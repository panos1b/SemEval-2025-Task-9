import os
import math
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup

# ----------------------------
# 1. Dataset and Preprocessing
# ----------------------------

class RecallDataset(Dataset):
    def __init__(self, texts, hazard_labels, product_labels, tokenizer, max_length=256):
        self.texts = texts
        self.hazard_labels = hazard_labels
        self.product_labels = product_labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['hazard'] = torch.tensor(self.hazard_labels[idx], dtype=torch.long)
        item['product'] = torch.tensor(self.product_labels[idx], dtype=torch.long)
        return item

# ----------------------------
# 2. Model Definition
# ----------------------------

class RecallClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_hazard_labels, num_product_labels, dropout_rate=0.3):
        super(RecallClassifier, self).__init__()
        # Load pretrained BERT encoder
        self.encoder = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.encoder.config.hidden_size

        # A dropout layer
        self.dropout = nn.Dropout(dropout_rate)

        # Two classification heads
        self.hazard_classifier = nn.Linear(hidden_size, num_hazard_labels)
        self.product_classifier = nn.Linear(hidden_size, num_product_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use the CLS token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        hazard_logits = self.hazard_classifier(cls_output)
        product_logits = self.product_classifier(cls_output)
        return hazard_logits, product_logits

# ----------------------------
# 3. Training and Evaluation Functions
# ----------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    running_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch}", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hazard_labels = batch['hazard'].to(device)
        product_labels = batch['product'].to(device)

        hazard_logits, product_logits = model(input_ids, attention_mask)

        loss_hazard = nn.CrossEntropyLoss()(hazard_logits, hazard_labels)
        loss_product = nn.CrossEntropyLoss()(product_logits, product_labels)
        loss = loss_hazard + loss_product

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    return avg_loss

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_hazard_preds = []
    all_product_preds = []
    all_hazard_labels = []
    all_product_labels = []
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hazard_labels = batch['hazard'].to('cpu').numpy()
        product_labels = batch['product'].to('cpu').numpy()
        hazard_logits, product_logits = model(input_ids, attention_mask)
        hazard_preds = torch.argmax(hazard_logits, dim=1).to('cpu').numpy()
        product_preds = torch.argmax(product_logits, dim=1).to('cpu').numpy()

        all_hazard_preds.extend(hazard_preds)
        all_product_preds.extend(product_preds)
        all_hazard_labels.extend(hazard_labels)
        all_product_labels.extend(product_labels)

    # Compute macro F1 for each head
    f1_hazard = f1_score(all_hazard_labels, all_hazard_preds, average='macro')
    f1_product = f1_score(all_product_labels, all_product_preds, average='macro')
    avg_f1 = (f1_hazard + f1_product) / 2.0
    return avg_f1, f1_hazard, f1_product

# ----------------------------
# 4. Main Worker for Distributed Training
# ----------------------------

def main_worker(args):
    # For torchrun, LOCAL_RANK is set automatically.
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize distributed process group using NCCL backend
    dist.init_process_group(backend='nccl', init_method='env://')
    
    torch.manual_seed(42)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # ----------------------------
    # Data Loading and Preprocessing
    # ----------------------------

    # Only the master process (rank 0) prepares and broadcasts label mappings.
    if rank == 0:
        df = pd.read_csv("combined_set.csv")
        # Create stratification column: product_hazard combination
        df['prod_hazard'] = df['product'].astype(str) + "_" + df['hazard'].astype(str)
        
        # Filter out unique occurrences in 'product' and 'hazard'
        df = df[
            df.duplicated(subset=['product'], keep=False) &
            df.duplicated(subset=['hazard'], keep=False)
        ]
        df = df[
            df.duplicated(subset=['prod_hazard'], keep=False)
        ]

        # Split train and validation (stratify by prod_hazard)
        train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['prod_hazard'])

        # Prepare label mappings
        hazard_classes = sorted(train_df['hazard'].unique().tolist())
        product_classes = sorted(train_df['product'].unique().tolist())
        hazard2idx = {label: idx for idx, label in enumerate(hazard_classes)}
        product2idx = {label: idx for idx, label in enumerate(product_classes)}
    else:
        train_df, val_df, hazard2idx, product2idx = None, None, None, None

    # Broadcast label mappings to all processes
    # Note: dist.broadcast_object_list expects a list.
    label_list = [hazard2idx, product2idx]
    dist.broadcast_object_list(label_list, src=0)
    hazard2idx, product2idx = label_list[0], label_list[1]

    # All processes load and split the data.
    if rank != 0:
        df = pd.read_csv("combined_set.csv")
        df['prod_hazard'] = df['product'].astype(str) + "_" + df['hazard'].astype(str)
                # Filter out unique occurrences in 'product' and 'hazard'
        df = df[
            df.duplicated(subset=['product'], keep=False) &
            df.duplicated(subset=['hazard'], keep=False)
        ]
        df = df[
            df.duplicated(subset=['prod_hazard'], keep=False)
        ]
        train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['prod_hazard'])

    # Map labels to integers for training/validation
    train_hazard_labels = train_df['hazard'].map(hazard2idx).tolist()
    train_product_labels = train_df['product'].map(product2idx).tolist()
    val_hazard_labels = val_df['hazard'].map(hazard2idx).tolist()
    val_product_labels = val_df['product'].map(product2idx).tolist()

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Create Dataset objects
    train_dataset = RecallDataset(
        texts=train_df['text'].tolist(),
        hazard_labels=train_hazard_labels,
        product_labels=train_product_labels,
        tokenizer=tokenizer
    )
    val_dataset = RecallDataset(
        texts=val_df['text'].tolist(),
        hazard_labels=val_hazard_labels,
        product_labels=val_product_labels,
        tokenizer=tokenizer
    )

    # Create DistributedSamplers and DataLoaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    # ----------------------------
    # Model, Optimizer, Scheduler Setup
    # ----------------------------
    num_hazard_labels = len(hazard2idx)
    num_product_labels = len(product2idx)

    model = RecallClassifier("bert-base-uncased", num_hazard_labels, num_product_labels)
    model.to(device)
    
    # Wrap the model with DistributedDataParallel
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.1 * total_steps),
                                                num_training_steps=total_steps)

    best_val_f1 = -np.inf
    # For simplicity, we only store validation F1 history here.
    val_f1_history = []

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)  # ensure shuffling
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        val_avg_f1, val_f1_hazard, val_f1_product = evaluate(model, val_loader, device)

        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - "
                  f"Val F1: {val_avg_f1:.4f} (Hazard: {val_f1_hazard:.4f}, Product: {val_f1_product:.4f})")
            val_f1_history.append(val_avg_f1)
            # Save best model based on validation F1 score
            if val_avg_f1 > best_val_f1:
                best_val_f1 = val_avg_f1
                torch.save(model.module.state_dict(), "best_model.pt")

    # ----------------------------
    # After Training: Inference on incidents_set.csv
    # ----------------------------
    if rank == 0:
        # Plot validation performance over epochs
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, args.epochs + 1), val_f1_history, label="Validation F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1 Score Over Epochs")
        plt.legend()
        plt.savefig("performance.png")
        plt.close()

        # Load the best model for inference
        best_model = RecallClassifier("bert-base-uncased", num_hazard_labels, num_product_labels)
        best_model.load_state_dict(torch.load("best_model.pt", map_location='cpu'))
        best_model.to(device)
        best_model.eval()

        # Create dataset for incidents_set.csv
        incidents_df = pd.read_csv("incidents_set.csv")
        incidents_texts = incidents_df['text'].tolist()

        class IncidentDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=256):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                item = {key: val.squeeze(0) for key, val in encoding.items()}
                return item

        incident_dataset = IncidentDataset(incidents_texts, tokenizer)
        incident_loader = DataLoader(incident_dataset, batch_size=args.batch_size, shuffle=False)

        hazard_preds_all = []
        product_preds_all = []

        with torch.no_grad():
            for batch in tqdm(incident_loader, desc="Predicting Incidents", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hazard_logits, product_logits = best_model(input_ids, attention_mask)
                hazard_preds = torch.argmax(hazard_logits, dim=1).to('cpu').numpy()
                product_preds = torch.argmax(product_logits, dim=1).to('cpu').numpy()
                hazard_preds_all.extend(hazard_preds)
                product_preds_all.extend(product_preds)

        # Map prediction indices back to label names
        idx2hazard = {idx: label for label, idx in hazard2idx.items()}
        idx2product = {idx: label for label, idx in product2idx.items()}

        hazard_labels_pred = [idx2hazard[idx] for idx in hazard_preds_all]
        product_labels_pred = [idx2product[idx] for idx in product_preds_all]

        submission = pd.DataFrame({
            "index": incidents_df.index,
            "hazard": hazard_labels_pred,
            "product": product_labels_pred
        })
        submission.to_csv("submission.csv", index=False)
        print("Submission saved to submission.csv")

    dist.destroy_process_group()

# ----------------------------
# 5. Main Function and Argument Parsing
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=185, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    main_worker(args)

if __name__ == '__main__':
    main()

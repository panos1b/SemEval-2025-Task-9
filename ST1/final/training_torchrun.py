import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

MODEL = 'bert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 50
EPOCHS = 30
AUGMENTED = 'ChatGPT_augmentation.csv'

# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
print(f'Using device: {device}')

# ------ Data Loading and Preprocessing ------

# Load combined dataset
combined_df = pd.read_csv('combined_set.csv')
augmented_df = pd.read_csv(AUGMENTED)

combined_df = pd.concat([combined_df, augmented_df], ignore_index=True)


# Split with stratification on product-category only
train_df, val_df = train_test_split(
    combined_df,
    test_size=0.05,
    random_state=69,
    stratify=combined_df['product-category']
)

# Encode labels
hazard_encoder = LabelEncoder()
train_hazard = hazard_encoder.fit_transform(train_df['hazard-category'])
val_hazard = hazard_encoder.transform(val_df['hazard-category'])

product_encoder = LabelEncoder()
train_product = product_encoder.fit_transform(train_df['product-category'])
val_product = product_encoder.transform(val_df['product-category'])

# Save the label encoders (only on rank 0 to avoid multiple writes)
if dist.get_rank() == 0:
    with open('hazard_encoder.pkl', 'wb') as f:
        pickle.dump(hazard_encoder, f)
    with open('product_encoder.pkl', 'wb') as f:
        pickle.dump(product_encoder, f)
    print("Label encoders saved to hazard_encoder.pkl and product_encoder.pkl")

# Compute class weights
hazard_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_hazard), y=train_hazard), 
                             dtype=torch.float32).to(device)
product_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(train_product), y=train_product), 
                              dtype=torch.float32).to(device)

# ------ Dataset and DataLoader ------
class TextDataset(Dataset):
    def __init__(self, texts, hazard_labels, product_labels, tokenizer, max_len):
        self.texts = texts
        self.hazard_labels = hazard_labels
        self.product_labels = product_labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'hazard': torch.tensor(self.hazard_labels[idx], dtype=torch.long),
            'product': torch.tensor(self.product_labels[idx], dtype=torch.long)
        }

# Use pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL)

train_dataset = TextDataset(
    train_df['text'].values,
    train_hazard,
    train_product,
    tokenizer,
    MAX_LEN
)

val_dataset = TextDataset(
    val_df['text'].values,
    val_hazard,
    val_product,
    tokenizer,
    MAX_LEN
)

train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, 
                         num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, 
                       num_workers=4, pin_memory=True)

# ------ Model Definition ------
class MultiTaskBERT(nn.Module):
    def __init__(self, n_hazard, n_product):
        super().__init__()

        # Use pretrained transformer
        self.bert = AutoModel.from_pretrained(MODEL)

        # Shared layers
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, 1200)
        self.batch_norm = nn.BatchNorm1d(1200)
        self.relu = nn.ReLU()

        # Hazard layer - Direct classification
        self.hazard_classifier = nn.Linear(1200, n_hazard)

        # Prodcut layer - Direct classification
        self.product_classifier = nn.Linear(1200, n_product)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.hidden_layer(pooled_output)
        x = self.batch_norm(x)
        x = self.relu(x)
        return self.hazard_classifier(x), self.product_classifier(x)

model = MultiTaskBERT(
    n_hazard=len(hazard_encoder.classes_),
    n_product=len(product_encoder.classes_)
).to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# ------ Training Setup ------
optimizer = AdamW(model.parameters(), lr=2e-5, no_deprecation_warning=True)
hazard_loss_fn = nn.CrossEntropyLoss(weight=hazard_weights)
product_loss_fn = nn.CrossEntropyLoss(weight=product_weights)

best_val_f1 = 0
best_model_weights = None
train_f1s = []
val_f1s = []

# ------ Evaluation Function ------
def evaluate(loader):
    model.eval()
    hazards, products = [], []
    hazard_preds, product_preds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            h_pred, p_pred = model(input_ids, attention_mask)
            hazards.extend(batch['hazard'].cpu().numpy())
            products.extend(batch['product'].cpu().numpy())
            hazard_preds.extend(torch.argmax(h_pred, dim=1).cpu().numpy())
            product_preds.extend(torch.argmax(p_pred, dim=1).cpu().numpy())

    gathered_hazards = [None] * dist.get_world_size()
    gathered_products = [None] * dist.get_world_size()
    gathered_h_preds = [None] * dist.get_world_size()
    gathered_p_preds = [None] * dist.get_world_size()

    dist.all_gather_object(gathered_hazards, hazards)
    dist.all_gather_object(gathered_products, products)
    dist.all_gather_object(gathered_h_preds, hazard_preds)
    dist.all_gather_object(gathered_p_preds, product_preds)

    if dist.get_rank() == 0:
        all_hazards = []
        all_products = []
        all_h_preds = []
        all_p_preds = []
        for i in range(dist.get_world_size()):
            all_hazards.extend(gathered_hazards[i])
            all_products.extend(gathered_products[i])
            all_h_preds.extend(gathered_h_preds[i])
            all_p_preds.extend(gathered_p_preds[i])
        return (
            f1_score(all_hazards, all_h_preds, average='macro'),
            f1_score(all_products, all_p_preds, average='macro')
        )
    return None, None

# ------ Training Loop ------

for epoch in range(EPOCHS):
    train_loader.sampler.set_epoch(epoch)
    model.train()
    # Foward pass
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hazard_pred, product_pred = model(input_ids, attention_mask)

        # Compute loss
        hazard_loss = hazard_loss_fn(hazard_pred, batch['hazard'].to(device))
        product_loss = product_loss_fn(product_pred, batch['product'].to(device))
        total_loss = hazard_loss + product_loss

        # Backpropagation
        total_loss.backward()
        optimizer.step()

    train_hazard_f1, train_product_f1 = evaluate(train_loader)
    val_hazard_f1, val_product_f1 = evaluate(val_loader)

    # Only on main script
    if dist.get_rank() == 0:
        # Calculate F1 scores - NOT COMPARABLE TO SEMEVAL F1 THIS IS FOR TRAINING!
        train_f1 = (train_hazard_f1 + train_product_f1) / 2
        val_f1 = (val_hazard_f1 + val_product_f1) / 2

        # Keep best model
        if val_f1 > best_val_f1:
            print(f'New best model at epoch {epoch+1} with F1: {val_f1:.4f}')
            best_val_f1 = val_f1
            best_model_weights = model.module.state_dict().copy()
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}')
        print(f'Train Hazard F1: {train_hazard_f1:.4f} | Val Hazard F1: {val_hazard_f1:.4f}')
        print(f'Train Product F1: {train_product_f1:.4f} | Val Product F1: {val_product_f1:.4f}')
        print('-' * 50)

# ------ Post-Training Processing ------
if dist.get_rank() == 0:
    model.module.load_state_dict(best_model_weights)
    torch.save(model.module.state_dict(), 'best_model.pth')
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_f1s, label='Training F1')
    plt.plot(val_f1s, label='Validation F1')
    plt.title('Model Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Average F1 Score')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    plt.close()

    # ------ Prediction on Incidents Set ------
    class InferenceDataset(Dataset):
        def __init__(self, texts, tokenizer, max_len):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

    # Load incidents data
    incidents_df = pd.read_csv('incidents_set.csv')
    index_col = incidents_df['Unnamed: 0'].values
    texts = incidents_df['text'].values

    # Create dataset and loader
    incident_dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    incident_loader = DataLoader(incident_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Predict
    model.eval()
    hazard_preds = []
    product_preds = []

    with torch.no_grad():
        for batch in incident_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            h_pred, p_pred = model(input_ids, attention_mask)
            hazard_preds.extend(torch.argmax(h_pred, dim=1).cpu().numpy())
            product_preds.extend(torch.argmax(p_pred, dim=1).cpu().numpy())

    # Decode predictions
    hazard_labels = hazard_encoder.inverse_transform(hazard_preds)
    product_labels = product_encoder.inverse_transform(product_preds)

    # Create submission file
    submission_df = pd.DataFrame({
        'Unnamed: 0': index_col,
        'hazard-category': hazard_labels,
        'product-category': product_labels
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

dist.destroy_process_group()
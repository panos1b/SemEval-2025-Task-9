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
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Params
MODEL = 'bert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 90
EPOCHS = 30
AUGMENTED = 'ChatGPT_augmentation.csv'
AUGMENTED_2 = 'ChatGPT_augmentation_2.csv'

# Change from ST1
class F1Loss(torch.nn.Module):
    """
    Aproximation of f1 as loss function
    """
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)
        y_true_one_hot = torch.nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).float()

        tp = (y_pred * y_true_one_hot).sum(dim=0)
        fp = ((1 - y_true_one_hot) * y_pred).sum(dim=0)
        fn = (y_true_one_hot * (1 - y_pred)).sum(dim=0)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        return 1 - f1.mean()


# Change from ST1
class HybridF1CrossEntropyLoss(torch.nn.Module):
    """
    This creates a loss function that is a balance between F1 aprox and Cross entropy
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.f1_loss = F1Loss()

    def forward(self, y_pred, y_true):
        ce_loss = self.cross_entropy(y_pred, y_true)
        f1_loss = self.f1_loss(y_pred, y_true)
        return self.alpha * ce_loss + (1 - self.alpha) * f1_loss


# Initialize distributed training
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')
print(f'Using device: {device}')

# ------- Data Loading and Preprocessing -------

# Read the CSV file
combined_df = pd.read_csv('combined_set.csv')
augmented_df = pd.read_csv(AUGMENTED)
augmented_2_df = pd.read_csv(AUGMENTED_2)

combined_df = pd.concat([combined_df, augmented_df], ignore_index=True)

# Change from ST1 - Create a new column that combines 'product' and 'hazard'
combined_df['stratify_col'] = combined_df['product'].astype(str) + "_" + combined_df['hazard'].astype(str)

# Filter out unique occurrences in 'product' and 'hazard'
combined_df = combined_df[
    combined_df.duplicated(subset=['product'], keep=False) &
    combined_df.duplicated(subset=['hazard'], keep=False)
]

# Make sure nothing got through
combined_df = combined_df[
    combined_df.duplicated(subset=['stratify_col'], keep=False)
]

# Perform stratified split
train_df, val_df = train_test_split(
    combined_df,
    test_size=0.3, # Change from ST1
    random_state=69,
    stratify=combined_df['stratify_col']  # Change from ST1 - Stratify on combined column
)



# Encode labels
# Change from ST1 - label is pre fitted then used for transformation. This is due to class nonexistence
hazard_encoder = LabelEncoder()
hazard_encoder = hazard_encoder.fit(combined_df['hazard'])
train_hazard = hazard_encoder.transform(train_df['hazard'])
val_hazard = hazard_encoder.transform(val_df['hazard'])

product_encoder = LabelEncoder()
product_encoder = product_encoder.fit(combined_df['product'])
train_product = product_encoder.transform(train_df['product'])
val_product = product_encoder.transform(val_df['product'])

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

# ------- Dataset and DataLoader -------
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

# Change from ST1 - Used more workers
train_sampler = DistributedSampler(train_dataset, shuffle=True)
val_sampler = DistributedSampler(val_dataset, shuffle=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=32, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                        num_workers=32, pin_memory=True)

# ------- Model Definition -------
class MultiTaskBERT(nn.Module):
    def __init__(self, n_hazard, n_product):
        super().__init__()

        # Use pretrained transformer
        self.bert = AutoModel.from_pretrained(MODEL)

        # Shared layers
        # Change from ST1 - Added dropout, changed dense layer size
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, 2304)
        self.batch_norm = nn.BatchNorm1d(2304)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Hazard branch
        # Change from ST1 - Hazard no longer classified by a single layer
        self.hazard_dense = nn.Linear(2304, 512)
        self.hazard_activation = nn.ReLU()
        self.hazard_dropout = nn.Dropout(0.45)
        self.hazard_batch_norm = nn.LayerNorm(512)
        self.hazard_classifier = nn.Linear(512, n_hazard)

        # Product branch
        # Change from ST1 - Product no longer classified by a single layer - Leaky ReLU
        self.product_dense = nn.Linear(2304, 1536)
        self.product_activation = nn.LeakyReLU(0.15)
        self.product_batch_norm = nn.LayerNorm(1536)
        self.product_classifier = nn.Linear(1536, n_product)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token output

        x = self.hidden_layer(pooled_output)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Change from ST1 - Same as in initialised model
        hazard_x = self.hazard_dense(x)
        hazard_x = self.hazard_activation(hazard_x)
        hazard_x = self.hazard_dropout(hazard_x)
        hazard_x = self.hazard_batch_norm(hazard_x)
        hazard_output = self.hazard_classifier(hazard_x)

        # Change from ST1 - Same as in initialised model
        product_x = self.product_dense(x)
        product_x = self.product_batch_norm(product_x)
        product_x = self.product_activation(product_x)
        product_output = self.product_classifier(product_x)


        return hazard_output, product_output

model = MultiTaskBERT(
    n_hazard=len(hazard_encoder.classes_),
    n_product=len(product_encoder.classes_)
).to(device)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# ------- Training Setup -------
# Change from ST1 increased learning rate for optimizer
optimizer = AdamW(model.parameters(), lr=2e-4)

# Change from ST1 - custom loss function
hazard_loss_fn = HybridF1CrossEntropyLoss(alpha=0.5)
product_loss_fn = HybridF1CrossEntropyLoss(alpha=0.5)

# Change from ST1 - This function is old and deprecated but adding the "cuda"
# argument was causing me issues
scaler = GradScaler()

best_val_f1 = 0
best_model_weights = None
train_f1s = []
val_f1s = []

# ------- Evaluation Function -------
def evaluate(loader):
    model.eval()
    hazards, products = [], []
    hazard_preds, product_preds = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with autocast():
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

        all_hazards = np.array(all_hazards)
        all_products = np.array(all_products)
        all_h_preds = np.array(all_h_preds)
        all_p_preds = np.array(all_p_preds)

        return (
            f1_score(all_hazards, all_h_preds, average='macro'),
            # Semeval f1 score
            f1_score(
                all_products[all_h_preds == all_hazards],
                all_p_preds[all_h_preds == all_hazards],
                average='macro'
            )
        )

    return None, None

# ------- Training Loop -------

for epoch in range(EPOCHS):
    train_loader.sampler.set_epoch(epoch)
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        hazard_labels = batch['hazard'].to(device)
        product_labels = batch['product'].to(device)

        # Change from ST1
        #
        # Autocast enables automatic mixed precision
        # this reduces memory usage and speeds up training
        # on GPUs with tensor cores
        # Inspiration drawn from the pytorch automatic mixed precision recipe
        #
        # Use our custom loss function
        with autocast():
            hazard_pred, product_pred = model(input_ids, attention_mask)
            hazard_loss = hazard_loss_fn(hazard_pred, hazard_labels)
            product_loss = product_loss_fn(product_pred, product_labels)
            total_loss = hazard_loss + product_loss

        # Change from ST1 - scaling needed due to worse precision of autocast which can push
        # gradients to being too small
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_hazard_f1, train_product_f1 = evaluate(train_loader)
    val_hazard_f1, val_product_f1 = evaluate(val_loader)

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

# ------- Post-Training Processing -------
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

    # ------- Prediction on Incidents Set -------
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
    incident_loader = DataLoader(incident_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

    # Predict
    model.eval()
    hazard_preds = []
    product_preds = []

    with torch.no_grad():
        for batch in incident_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with autocast():
                h_pred, p_pred = model(input_ids, attention_mask)
            hazard_preds.extend(torch.argmax(h_pred, dim=1).cpu().numpy())
            product_preds.extend(torch.argmax(p_pred, dim=1).cpu().numpy())

    # Decode predictions
    hazard_labels = hazard_encoder.inverse_transform(hazard_preds)
    product_labels = product_encoder.inverse_transform(product_preds)

    # Create submission file
    submission_df = pd.DataFrame({
        'Unnamed: 0': index_col,
        'hazard': hazard_labels,
        'product': product_labels
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

dist.destroy_process_group()
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Local inference script for the MultiTaskBERT model (hazard-category and product-category).

This script:
  - Loads saved model weights from 'best_model.pth'
  - Loads label encoder files 'hazard_encoder.pkl' and 'product_encoder.pkl'
  - Loads the incidents data from 'incidents_set.csv'
  - Runs inference on the data using the model (without distributed training)
  - Creates a submission CSV file named 'submission.csv'
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# --- Constants ---
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512
BATCH_SIZE = 50  # Same as training batch size, adjust if needed

# --- Model Definition ---
class MultiTaskBERT(nn.Module):
    def __init__(self, n_hazard, n_product, model_name=MODEL_NAME):
        super(MultiTaskBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_layer = nn.Linear(self.bert.config.hidden_size, 1200)
        self.batch_norm = nn.BatchNorm1d(1200)
        self.relu = nn.ReLU()
        self.hazard_classifier = nn.Linear(1200, n_hazard)
        self.product_classifier = nn.Linear(1200, n_product)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token output
        x = self.hidden_layer(pooled_output)
        x = self.batch_norm(x)
        x = self.relu(x)
        hazard_logits = self.hazard_classifier(x)
        product_logits = self.product_classifier(x)
        return hazard_logits, product_logits

# --- Inference Dataset ---
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
        # Remove the extra batch dimension added by return_tensors='pt'
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }

# --- Main Inference Function ---
def main():
    # Set device: use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load the label encoders
    with open('hazard_encoder.pkl', 'rb') as f:
        hazard_encoder = pickle.load(f)
    with open('product_encoder.pkl', 'rb') as f:
        product_encoder = pickle.load(f)

    n_hazard = len(hazard_encoder.classes_)
    n_product = len(product_encoder.classes_)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Initialize the model and load saved weights
    model = MultiTaskBERT(n_hazard, n_product, model_name=MODEL_NAME)
    state_dict = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Load and Prepare the Incidents Data ---
    incidents_df = pd.read_csv('incidents_set.csv')
    # Use the column "Unnamed: 0" if it exists, otherwise use the DataFrame index
    if 'Unnamed: 0' in incidents_df.columns:
        index_col = incidents_df['Unnamed: 0'].values
    else:
        index_col = incidents_df.index.values
    texts = incidents_df['text'].values

    # Create dataset and DataLoader (num_workers=0 for single-threaded loading)
    dataset = InferenceDataset(texts, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    hazard_preds = []
    product_preds = []

    # Run inference without gradient tracking
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            h_logits, p_logits = model(input_ids, attention_mask)
            h_pred = torch.argmax(h_logits, dim=1)
            p_pred = torch.argmax(p_logits, dim=1)
            hazard_preds.extend(h_pred.cpu().numpy())
            product_preds.extend(p_pred.cpu().numpy())

    # Decode predictions back to their original string labels
    hazard_labels = hazard_encoder.inverse_transform(np.array(hazard_preds))
    product_labels = product_encoder.inverse_transform(np.array(product_preds))

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'Unnamed: 0': index_col,
        'hazard-category': hazard_labels,
        'product-category': product_labels
    })
    submission_df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")

if __name__ == "__main__":
    main()

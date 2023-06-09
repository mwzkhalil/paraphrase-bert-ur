import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import UrduParaphraseDataset
from model import ParaphraseModel

# Set random seed for reproducibility
torch.manual_seed(42)

# Load Urdu CSV dataset
csv_path = 'urdu_dataset.csv'
df = pd.read_csv(csv_path)

# Split dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define training parameters
epochs = 10
batch_size = 16
learning_rate = 2e-5

# Create data loaders
train_dataset = UrduParaphraseDataset(train_df, tokenizer, max_length=128)
val_dataset = UrduParaphraseDataset(val_df, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Train the model
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        loss = criterion(logits.view(-1, logits.shape[-1]), input_ids.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}')

# Save the trained model
torch.save(model.state_dict(), 'paraphrase_model.pt')

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from torch.nn import functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
import math
nltk.download('punkt')



# PREPROCESS
# Load data
data = pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=['review', 'label'])

# Tokenization
data['tokens'] = data['review'].apply(word_tokenize)

# Creating vocabulary
word_counts = Counter(word for tokens in data['tokens'] for word in tokens)
vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common())}
vocab['<pad>'] = 0

# Encoding and padding
max_len = max(map(len, data['tokens']))
encoded_reviews = [[vocab[word] for word in tokens] + [0] * (max_len - len(tokens)) for tokens in data['tokens']]
labels = data['label'].values

# Splitting the dataset
train_data, test_data, train_labels, test_labels = train_test_split(encoded_reviews, labels, test_size=0.2, random_state=42)

# Dataloaders
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        return torch.tensor(self.reviews[idx]), torch.tensor(self.labels[idx])

train_dataset = ReviewDataset(train_data, train_labels)
test_dataset = ReviewDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# TRANSFORMER MODEL
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim=1)  # Change to mean across the correct dimension
        output = self.decoder(output)
        return output.squeeze()  # Squeeze to ensure it matches target size


# TRAINING AND TESTING
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for text, labels in train_loader:
        text, labels = text.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(text)
        loss = criterion(output, labels.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for text, labels in test_loader:
            text, labels = text.to(device), labels.to(device)
            output = model(text)
            loss = criterion(output, labels.float())
            total_loss += loss.item()
            predicted = (output > 0.5).int()
            correct += (predicted == labels).sum().item()
    return total_loss / len(test_loader), correct / len(test_loader.dataset)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  # the size of vocabulary
d_model = 128  # embedding dimension
nhead = 8  # number of heads
nhid = 512  # dimension of the feedforward network model
nlayers = 128
dropout = 0.2

model = TransformerModel(ntokens, d_model, nhead, nhid, nlayers, dropout).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(20):  # num epochs
    train_loss = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, test_loader, criterion, device)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')


def chatbot():
    model.eval()
    while True:
        review = input("Enter a review (or 'quit' to stop): ")
        if review.lower() == 'quit':
            break
        tokens = [vocab.get(word, 0) for word in word_tokenize(review)]
        if len(tokens) < max_len:
            tokens += [0] * (max_len - len(tokens))
        tokens = torch.tensor(tokens).unsqueeze(0).to(device)
        prediction = model(tokens)
        print("Positive" if prediction.item() > 0.5 else "Negative")

chatbot()

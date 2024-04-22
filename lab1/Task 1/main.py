import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_sequence
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# PREPROCESS
# load and prepare data uncomment your line and keep comment the other ones.

data = pd.read_csv('amazon_cells_labelled.txt', sep='\t', names=['review', 'label'])  #Nabez esecution
#data = pd.read_csv(r'C:\Users\39393\Desktop\UNIPD\ERASMUS+\DEEP LEARNING\d7047e\lab1\Task 1\amazon_cells_labelled.txt', sep='\t', names=['review', 'label'])  #Francesco esecution

# checking how many pos and neg reviews
review_counts = data['label'].value_counts()
print(f'Count of reviews by sentiment: {review_counts}')

nltk.download('punkt')
data['tokens'] = data['review'].apply(word_tokenize)

# create a vocabulary and map tokens to indices
vocab = {word: idx for idx, word in enumerate(set(word for review in data['tokens'] for word in review), 1)}
data['indexed'] = data['tokens'].apply(lambda x: [vocab[word] for word in x])

# pad sequences to a maximum length
max_len = max(len(review) for review in data['indexed'])
data['padded'] = data['indexed'].apply(lambda x: x + [0]*(max_len - len(x)))

# convert to tensord and split dataset
features = torch.tensor(data['padded'].tolist())
labels = torch.tensor(data['label'].tolist())

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
train_data = TensorDataset(train_features, train_labels)
test_data = TensorDataset(test_features, test_labels)
train_loader = DataLoader(train_data, batch_size=10)
test_loader = DataLoader(test_data, batch_size=10)

# CREATE A MODEL
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x).mean(dim=1)
        hidden = self.relu(self.fc1(embedded))
        output = self.fc2(hidden)
        return output

model = TextClassifier(len(vocab)+1, 50, 100, 2)

# TRAIN THE MODEL
criterion = nn.CrossEntropyLoss()
optimzier = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    for texts, labels in train_loader:
        optimzier.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimzier.step()
    print(f'Epoch {epoch+1}: Training Loss = {loss.item()}')

# TEST MODEL
# TEST MODEL
correct = 0
total = 0
predicted = []
actual = []
with torch.no_grad():
    for texts, labels in test_loader:
        outputs = model(texts)
        _, preds = torch.max(outputs.data, 1)  # Correct variable naming here

        # Convert Tensor to list and then extend
        predicted.extend(preds.tolist())
        actual.extend(labels.tolist())

print('Test Accuracy:', accuracy_score(actual, predicted)*100)
print('Classification Report:\n', classification_report(actual, predicted))
print('Confusion Matrix:\n', confusion_matrix(actual, predicted))
try:
    print('ROC AUC Score:', roc_auc_score(actual, predicted))
except ValueError as e:
    print(e)


# CHATBOT FUNCTIONS
def chatbot_respone(text):
    tokens = word_tokenize(text)
    indexed = [vocab.get(word, 0) for word in tokens]
    padded = indexed + [0]*(max_len - len(indexed))
    input_tensor = torch.tensor([padded])
    output = model(input_tensor)
    _, predicted = torch.max(output.data, 1)
    return "Positive" if predicted.item() == 1 else "Negative"

while True:
    user_input = input('Type your message: ')
    if user_input.lower() == 'quit':
        break
    response = chatbot_respone(user_input)
    print(f'The sentiment of the review is: {response}')
    




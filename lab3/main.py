'''
Insperation guides:
https://medium.com/@raman.shinde15/image-captioning-with-flickr8k-dataset-bleu-4bcba0b52926
https://thepythoncode.com/article/image-captioning-with-pytorch-and-transformers-in-python

Overall comments:
Image captioning uses one to many RNN's.

About Flickr 8k dataset
Images: Contains a total of 8092 jpg format with different shapes and sizes.
        6000 for train, 1000 for test, 1000 for development
Captions.txt Contains 5 captions for each image, total of 40460 captions.

Size of training vocabulary: 7371

Architecture
We will use CNN + LSTM with attention.
CNN: To extract features from the image.
LSTM: To generate a description from the extracted information of the image

Note:
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel
RGB images of shape (3 x H x W), where H and W are expected to be at least 299. The images have to be loaded in
to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
'''
import os
import pandas as pd
import spacy

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split

from PIL import Image
from torchvision.transforms import transforms

import torch.nn as nn
import torch

import torchvision.models as models
import torch.optim as optim

from torch import optim

from nltk.translate.bleu_score import corpus_bleu
import matplotlib.pyplot as plt

from nltk.translate.meteor_score import meteor_score
from rouge import Rouge


# Custom Dataset for text
'''
We want to convert text -> numerical values
1. We need a Vocabulary mapping each word to a index (to convert the string to numerical value)
2. We need to setup a pytorch dataset to load the data
3. Setup padding of every batch (all examples should be of same seq_length in the batch and setup a dataloader)
'''
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        # the freq_threshold it to collect the most often used words
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    #  Example:
    #  Converts "I love peanuts" -> ["i", "love", "peanuts"]

    def build_vocabulary(self, sentence_list):  # we're getting a list of all the captions in this sentence list
        frequencies = {}
        # here we are going through all the captions to count how many times is a specific word repeats
        # if it is above the threshold (5) then we are going to include it
        idx = 4  # we start at idx=4 because self.itos has 4 keys/values pairs already

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        # converts text to numerical values
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text

            # if token is inside stoi dict then it has surpassed the threshold then
            # we are going to add that idx to the token
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        super(Dataset).__init__()
        '''
        The purpose of this class is to load 
        '''
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df['image']
        self.captions = self.df['caption']

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):  # to see the length of the dataframes
        return len(self.df)

    def __getitem__(self, index):  # to get a single example
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert('RGB')

        #print(f' Loaded captions: {caption}')

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # telling the model this is the start of the sentence
        numericalized_caption += self.vocab.numericalize(caption)  # converting each word to an index in our vocabulary
        numericalized_caption.append(self.vocab.stoi["<EOS>"])  # telling the model this is the end of the sentence

        return img, torch.tensor(numericalized_caption)


# When we use sequence mocdels, it is very important that the sequence lengths are all the same for the entire batch
# We are using captions, and all captions will be different length
# class MyCollate will check what is the longest length that is in our batch and PAD everything to that specific length
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]  # create extra dimension
        imgs = torch.cat(imgs, dim=0)  # concatenating all the images across the dim=0
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True).features

        # Freeze the parameters if not training the CNN
        if not train_CNN:
            for param in self.vgg16.parameters():
                param.requires_grad = False

        # Adaptive pooling to consistently produce a fixed size output
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(512 * 7 * 7, embed_size)  # Correctly calculate the input dimension
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        #print(f"Input shape encoder: {images.shape}")
        features = self.vgg16(images)
        #print(f"After VGG16 encoder: {features.shape}")
        features = self.avgpool(features)
        #print(f"After AvgPool encoder: {features.shape}")
        features = features.view(-1, 512 * 7 * 7)  # Correct the flattening
        #print(f"After Flatten encoder: {features.shape}")
        features = self.fc(features)
        #print(f"After FC encoder: {features.shape}")
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        #print(f'Caption decoder: {captions.shape}')
        #print(f'Feature decoder: {features.shape}')
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        #print(f'embedding decoder: {embeddings.shape}')
        hiddens, _ = self.lstm(embeddings)
        #print(f'hiddens decoder: {hiddens.shape}')
        outputs = self.linear(hiddens)
        #print(f'outputs decoder: {outputs.shape}')
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        #print(f'features CNN2RNN:{features.shape}')
        outputs = self.decoderRNN(features, captions)
        #print(f'output CNN2RNN:{outputs.shape}')
        return outputs

    def caption_image(self, image, vocabulary, max_length=20):
        result_caption = []


        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                #print("Initial x from CNN:", x)
                #print(f"Hidden State: {hiddens.shape}")
                #print(f"Cell State:{states.shape}") # state size = [num_layers * num_directions, batch_size, hidden_size]
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                #print("Output from Linear:", output)
                predicted = output.argmax(1)
                predicted_word = vocabulary.itos[predicted.item()]
                #print(f"Predicted word: {predicted_word}")

                result_caption.append(predicted_word)

                if predicted_word == "<EOS>":
                    states = None
                    break

                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                final_caption = ' '.join(result_caption)
                final_caption = final_caption.replace('<SOS>', '')

            return final_caption



img_path = 'flickr 8k/Images'
caption_path = 'flickr 8k/captions.txt'

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_loader(dataset, batch_size=64, num_workers=2, shuffle=True, pin_memory=True):
    if isinstance(dataset, torch.utils.data.Subset):
        pad_idx = dataset.dataset.vocab.stoi["<PAD>"]
    else:
        pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx)
    )
    return loader


def create_splits(dataset, split_ratios=(0.8, 0.1, 0.1)):
    total = len(dataset)
    train_size = int(split_ratios[0] * total)
    val_size = int(split_ratios[1] * total)
    test_size = total - train_size - val_size
    print(f'Sizes - train: {train_size}, val: {val_size}, test: {test_size} ')
    return random_split(dataset, [train_size, val_size, test_size])

def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print("Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def show_images(images, vocab, captions):
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i+1)
        plt.imshow(img.cpu().permute(1, 2, 0).numpy())
        plt.title(captions)
        plt.axis('off')
    plt.show()

def train_and_validate(models, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0
        if epoch % 3 == 0:
            checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        # Training loop
        for idx, (imgs, captions) in enumerate(train_loader):
            #print(f"After iteration {idx}, Shape: {imgs.shape}, {captions.shape}")
            imgs, captions = imgs.to(device), captions.to(device)

            # Forward pass
            outputs = models(imgs, captions[:-1])  # Exclude the <EOS> token for input
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
            # mask the pad token
            #pad_token_id = 0
            #mask= (captions != pad_token_id)
            #loss *= mask.float()
            #loss = loss.sum() / mask.sum()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            if idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx}/{len(train_loader)}], Loss: {loss.item()}")

        # Calculate average training loss per epoch
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss Epoch {epoch+1}: {avg_train_loss}")

        # Validation loop
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for imgs, captions in val_loader:
                imgs, captions = imgs.to(device), captions.to(device)
                outputs = model(imgs, captions[:-1])
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
                total_val_loss += loss.item()

        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Validation Loss Epoch {epoch+1}: {avg_val_loss}")


def test_model(model, test_loader, vocabulary):
    model.eval()
    references = []
    hypotheses = []

    with torch.no_grad():
        for images, all_captions in test_loader:
            images = images.to(device)
            all_captions = all_captions.to(device)

            for i in range(images.size(0)):  # Loop through each image in the batch
                image = images[i]
                hypothesis = model.caption_image(image, vocabulary)
                # Split the generated caption into tokens and make sure it's a list of strings
                hypothesis_tokens = hypothesis.split()
                hypotheses.append(hypothesis_tokens)  # Directly append the list of tokens

                # Retrieve the corresponding captions for the current image
                image_captions = all_captions[:, i]  # Select the ith column for all caption sequences
                image_references = []

                for idx in image_captions:
                    if idx.item() not in {vocabulary.stoi["<SOS>"], vocabulary.stoi["<EOS>"], vocabulary.stoi["<PAD>"]}:
                        word = vocabulary.itos[idx.item()]
                        image_references.append(word)

                references.append([' '.join(image_references).split()])  # A list containing a single list of words

                if i < 1:  # Print for the first image only for checking
                    print(f"Generated Caption: {' '.join(hypothesis_tokens)}")
                    print(f"Reference Captions: {' '.join(image_references)}")

        # Compute BLEU scores or other metrics outside the loop
        bleu_score = corpus_bleu(references, hypotheses, weights=(1,0,0,0))
        print(f"BLEU score on the test set: {bleu_score:.4f}")
'''
 BLEU might not perfectly capture the quality of image captions if the generated text deviates in structure from the 
 reference but still correctly describes the image. Consider using other metrics like METEOR, ROUGE, 
'''


if __name__ == '__main__':
    full_dataset = FlickrDataset(img_path, caption_path, transform=transform)
    train_dataset, val_dataset, test_dataset = create_splits(full_dataset) # captions are displayed as text here.

    # img = torch.randn(1, 3, 299, 299)  # Include batch dimension in the test tensor
    # model = EncoderCNN(embed_size=256, train_CNN=False)
    # model.eval()
    # try:
    #     output = model(img)
    # except Exception as e:
    #     print(e)

    train_loader = get_loader(train_dataset, batch_size=64, shuffle=True)
    val_loader = get_loader(val_dataset, batch_size=64, shuffle=False)
    test_loader = get_loader(test_dataset, batch_size=64, shuffle=False)

    #for images, captions in train_loader:
        #print(f"Train Batch captions: {captions}")
    #for images, captions in val_loader:
        #print(f"Val Batch captions: {captions}")
    #for images, captions in test_loader:
        #print(f"Test Batch captions: {captions}")

    print('Loaders are ready for training, validation, and testing.')

    embed_size = 256
    hidden_size = 512
    vocab_size = len(full_dataset.vocab)
    print(vocab_size)
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 4
    batch_size = 32
    load_model = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Model setup
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=full_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load('my_checkpoint.pth.tar'))

    #train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs, device)
    test_model(model, test_loader, full_dataset.vocab)
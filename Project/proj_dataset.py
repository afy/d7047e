import pandas as pd
import nltk
from nltk import word_tokenize
import torch
import re
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# training: even sample from SOLID
# val: random rample from SOLID
# te: OLID, HASOC

FOLDERSPATH = r'datasets'
SOLID_HARDLINECAP = 100_000


import warnings
warnings.warn("Disabling panda dataframe warnings, to enable comment out lines below")
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def _load_set(data):        
    #review_counts = data['label1'].value_counts()
    #print(f'Count of reviews by sentiment: {review_counts}')
    data['tokens'] = data['sentence'].apply(word_tokenize)
    vocab = {word: idx for idx, word in enumerate(set(word for sentence in data['tokens'] for word in sentence), 1)}
    data['indexed'] = data['tokens'].apply(lambda x: [vocab[word] for word in x])
    max_len = max(len(sentence) for sentence in data['indexed'])
    data['padded'] = data['indexed'].apply(lambda x: x + [0]*(max_len - len(x)))
    features = torch.tensor(data['padded'].tolist())
    labels = torch.tensor(data['label1'].tolist())
    return TensorDataset(features, labels), vocab


def _clean_tweet(tweet):
    tweet += " "
    #tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)  # Remove @ mentions
    tweet = re.sub(r'\@(.*?)\s', '@USER ', tweet) # Replace @ mentions -> @USER
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)  # Remove URLs
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)  # Only keep text characters
    tweet = re.sub(r" +", ' ', tweet)  # Remove multiple spaces
    return tweet.strip()


def _preprocess(df):
    for index, row in df.iterrows():
        df.at[index, 'sentence'] = _clean_tweet(row["sentence"])
    return df


def _load_olid_test_set():
    olid_path = FOLDERSPATH + r'\\OLID\\OLID_Tain.txt'
    data = pd.read_csv(olid_path, sep='\t', names=['id','sentence', 'label1','label2', 'label3'])  #Francesco esecution 
    data = data.drop(axis=1, labels = ['id','label2','label3'])
    data = data.drop(axis=0, index=0)
    print(f"Loaded {data.shape[0]} lines from {olid_path}")
    for index, row in data.iterrows():
        if row['label1'] == "OFF":
            data.at[index, 'label1'] = 0 
        else:
            data.at[index, 'label1'] = 1
    return data


def _load_ds2_file(hasoc_path, path):
    data = pd.read_csv(hasoc_path + path, sep='\t', names=['id','sentence', 'label1','label2', 'label3'])
    data = data.drop(index=0, axis=0)
    data = data.drop(columns=['id','label2','label3'], axis=1)
    for index, row in data.iterrows():
        if row['label1'] == "HOF":
            data.at[index, 'label1'] = 0 
        else:
            data.at[index, 'label1'] = 1
    data['label1'] = data['label1'].astype(int)
    print(f"Loaded {data.shape[0]} lines from {path}")
    return data


def _load_hasoc_test_set():
    fp = FOLDERSPATH + r'\\HASOCData\\'
    file1 = r'english_dataset.tsv'
    file2 = r'hasoc2019_en_test-2919.tsv'
    data = pd.concat([_load_ds2_file(fp, file1), _load_ds2_file(fp, file2)])
    return _preprocess(data)


def _load_ds3_file(fp, path, nrows, capsize):
    data = pd.read_excel(
        fp + path, 
        names=['sentence','label1'],
        nrows=nrows
    )
    print(f"Loaded {data.shape[0]} lines from {path} to ensure sufficient randomness, cap: {capsize}")
    data = data.sample(frac=1)
    data = data[:capsize]
    return data


def _load_solid_set(size, tr_split_perc):
    solid_path = FOLDERSPATH + r'\\OffenseEval\\'
    file_off = r'file_off.xlsx'
    file_not = r'file_not.xlsx'

    size = size//2
    assert size < 150_000, "Maximum size exceeded for size argument (SOLID loader)"

    d_off = _load_ds3_file(
            solid_path, 
            file_off, 
            nrows=SOLID_HARDLINECAP, 
            capsize=size)

    d_not = _load_ds3_file(
            solid_path, 
            file_not, 
            nrows=SOLID_HARDLINECAP, 
            capsize=size)
  
    p = int(tr_split_perc*size)
    tr_off, tr_not  = d_off.iloc[:p, :], d_not.iloc[:p, :]
    vl_off, vl_not  = d_off.iloc[p:, :], d_not.iloc[p:, :]

    tr = pd.concat([tr_off, tr_not]).sample(frac=1)
    vl = pd.concat([vl_off, vl_not]).sample(frac=1)
    return tr, vl


def get_loaders(train_size = 40_000, tr_split_percentage = 0.8, batch_size = 32):
    olid = _load_olid_test_set()
    hasoc = _load_hasoc_test_set()
    solid_tr, solid_vl = _load_solid_set(train_size, tr_split_percentage)

    nltk.download('punkt')
    tr_set, vt = _load_set(solid_tr)
    vl_set, vv = _load_set(solid_vl)
    te1_set, vt1 = _load_set(hasoc)
    te2_set, vt2 = _load_set(olid)

    train_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(vl_set, batch_size=batch_size)
    test2_loader = DataLoader(te1_set, batch_size=batch_size)
    test1_loader = DataLoader(te2_set, batch_size=batch_size)
    
    # combine separate vocabularies to create  
    vocab = dict(vt)
    vocab.update(vv)
    vocab.update(vt1)
    vocab.update(vt2)
    print(len(vt),len(vv),len(vt1),len(vt2),len(vocab))

    return train_loader, val_loader, test1_loader, test2_loader, vocab
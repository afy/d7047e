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

def _clean_tweet(tweet):
    tweet += " "
    #tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet)  # Remove @ mentions
    tweet = re.sub(r'\@(.*?)\s', '@USER ', tweet) # Replace @ mentions -> @USER
    tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet)  # Remove URLs
    tweet = re.sub(r"[^a-zA-Z.!?']", ' ', tweet)  # Only keep text characters
    tweet = re.sub(r" +", ' ', tweet)  # Remove multiple spaces
    return tweet.strip()

def _load_set(data, off_token="OFF", load_solid=False, cutoff=0.2):
    if not load_solid: #OLID/HASOC
        for index, row in data.iterrows():
            if row['label1'] == off_token:
                data.at[index, 'label1'] = 0 
            else:
                data.at[index, 'label1'] = 1
    else:
        for index, row in data.iterrows():
            data.at[index, 'label1'] = 1 if float(row['label1']) < cutoff else 0
        data['label1'] = data['label1'].astype(int)
            
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
    return data

def _load_ds2_file(hasoc_path, path):
    data = pd.read_csv(hasoc_path + path, sep='\t', names=['id','sentence', 'label1','label2', 'label3'])
    data = data.drop(index=0, axis=0)
    data = data.drop(columns=['id','label2','label3'], axis=1)
    print(f"Loaded {data.shape[0]} lines from {path}")
    return data

def _load_hasoc_test_set():
    fp = FOLDERSPATH + r'\\HASOCData\\'
    file1 = r'english_dataset.tsv'
    file2 = r'hasoc2019_en_test-2919.tsv'
    data = pd.concat([_load_ds2_file(fp, file1), _load_ds2_file(fp, file2)])
    return _preprocess(data)

def _load_ds3_file(fp, path, nrows):
    data = pd.read_excel(
        fp + r'task_a_part' + path + r'.xlsx', 
        names=['id','sentence','label1','label2'],
        nrows=nrows
    )
    print(f"Loaded {data.shape[0]} lines from {path}")
    return data

def _load_solid_set(size):
    solid_path = FOLDERSPATH + r'\\OffenseEval\\'
    if size> 1_300_000:
        data = _load_ds3_file(solid_path, "1")
        for i in range(2,2):
            data = pd.concat([data, _load_ds3_file(solid_path, str(i))])
    else:
        data = _load_ds3_file(solid_path, "1", nrows=size)
    data = data.drop(columns=['id'], axis=1)
    data = data.dropna(axis=0) # remove rows containing NaN 
    data = data[:size]
    return data

def get_loaders(train_size = 40_000, tr_split_percentage = 0.8, batch_size = 32, cutoff=0.2):
    olid = _load_olid_test_set()
    hasoc = _load_hasoc_test_set()
    solid = _load_solid_set(train_size)

    p = int(tr_split_percentage * solid.shape[0])
    solid_tr = solid.loc[:p,]
    solid_vl = solid.loc[p:,]

    nltk.download('punkt')
    tr_set, vt = _load_set(solid_tr, load_solid=True, cutoff=cutoff)
    vl_set, vv = _load_set(solid_vl, load_solid=True, cutoff=cutoff)
    te1_set, vt1 = _load_set(hasoc, off_token="HOF")
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
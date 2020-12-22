# torchtext
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import os
from os.path import join as osp
from os import listdir as odir
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import itertools
import random
#from IPython.display import display, HTML

import re
import nltk
from nltk import stem
nltk.download('punkt')

# nltkによる形態素エンジンを用意
def nltk_analyzer(text):
    stemmer = stem.LancasterStemmer()
    text = re.sub(re.compile(r'[!-\/:-@[-`{-~]'), ' ', text)
    text = stemmer.stem(text)
    text = text.replace('\n', '') # 改行削除
    text = text.replace('\t', '') # タブ削除
    morph = nltk.word_tokenize(text)
    return morph

if __name__ == "__main__":
    train_pos_dir = 'aclImdb/train/pos'
    train_neg_dir = 'aclImdb/train/neg'

    test_pos_dir = 'aclImdb/test/pos'
    test_neg_dir = 'aclImdb/test/neg'

    header = ['text', 'label', 'label_id']

    train_pos_files = odir(train_pos_dir)
    train_neg_files = odir(train_neg_dir)
    test_pos_files = odir(test_pos_dir)
    test_neg_files = odir(test_neg_dir)

    def make_row(root_dir, files, label, idx):
        row = []
        for file in files:
            tmp = []
            with open(osp(root_dir, file), 'r') as f:
                text = f.read()
                tmp.append(text)
                tmp.append(label)
                tmp.append(idx)
            row.append(tmp)
        return row

    row = make_row(train_pos_dir, train_pos_files, 'pos', 0)
    row += make_row(train_neg_dir, train_neg_files, 'neg', 1)
    train_df = pd.DataFrame(row, columns=header)

    row = make_row(test_pos_dir, test_pos_files, 'pos', 0)
    row += make_row(test_pos_dir, test_pos_files, 'neg', 1)
    test_df = pd.DataFrame(row, columns=header)

    extract_columns = ["text", "label_id"]

    #train_df[extract_columns].to_csv('train.tsv', index=False, header=None)
    #test_df[extract_columns].to_csv('test.tsv', index=False, header=None)

    train_df = pd.read_csv('train.tsv', header=None)
    print(train_df.head())

    imdb_dir = './'

    word_embedding_dir = './'

    TEXT = data.Field(sequential=True, tokenize=nltk_analyzer, lower=True, include_lengths=True, batch_first=True)
    LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

    train, test = data.TabularDataset.splits(
          path=imdb_dir, train='train.tsv', test='test.tsv', format='tsv',
          fields=[('Text', TEXT), ('Label', LABEL)])

    glove_vectors = Vectors(name=word_embedding_dir + "glove.6B.200d.txt")
    TEXT.build_vocab(train, vectors=glove_vectors, min_freq=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 100
    EMBEDDING_DIM = 200
    LSTM_DIM = 128
    VOCAB_SIZE = TEXT.vocab.vectors.size()[0]
    TAG_SIZE = 2
    DA = 64
    R = 3

    class BiLSTMEncoder(nn.Module):
        def __init__(self, embedding_dim, lstm_dim, vocab_size):
            super(BiLSTMEncoder, self).__init__()
            self.lstm_dim = lstm_dim
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

            self.word_embeddings.weight.data.copy_(TEXT.vocab.vectors)

            self.word_embeddings.requires_grad_ = False

            self.bilstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True, bidirectional=True)

        def forward(self, text):
            embeds = self.word_embeddings(text)

            out, _ = self.bilstm(embeds)

            return out


    class SelfAttention(nn.Module):
        def __init__(self, lstm_dim, da, r):
            super(SelfAttention, self).__init__()
            self.lstm_dim = lstm_dim
            self.da = da
            self.r = r
            self.main = nn.Sequential(
                    nn.Linear(lstm_dim*2, da),
                    nn.Tanh(),
                    nn.Linear(da, r)
            )

        def forward(self, out):
            return F.softmax(self.main(out), dim=1)

    
    class SelfAttentionClassifier(nn.Module):
        def __init__(self, lstm_dim, da, r, tagset_size):
            super(SelfAttentionClassifier, self).__init__()
            self.lstm_dim = lstm_dim
            self.r = r
            self.attn = SelfAttention(lstm_dim, da, r)
            self.main = nn.Linear(lstm_dim * 6, tagset_size)


        def forward(self, out):
            attention_weight = self.attn(out)
            m1 = (out * attention_weight[:,:, 0].unsqueeze(2)).sum(dim=1)
            m2 = (out * attention_weight[:,:, 1].unsqueeze(2)).sum(dim=1)
            m3 = (out * attention_weight[:,:, 2].unsqueeze(2)).sum(dim=1)

            feats = torch.cat([m1, m2, m3], dim=1)
            return F.log_softmax(self.main(feats)), attention_weight

    encoder = BiLSTMEncoder(EMBEDDING_DIM, LSTM_DIM, VOCAB_SIZE).to(device)
    classifier = SelfAttentionClassifier(LSTM_DIM, DA, R, TAG_SIZE).to(device)
    loss_function = nn.NLLLoss()

    optimizer = optim.Adam(chain(encoder.parameters(), classifier.parameters()), lr=0.01)

    train_iter, test_iter = data.Iterator.splits((train, test), batch_size=(BATCH_SIZE, BATCH_SIZE), device=device, repeat=False, sort=False)

    print("test")
    print(len(train_iter))


    losses = []
    for epoch in range(10):
        all_loss = 0

        for idx, batch in enumerate(train_iter):
            batch_loss = 0
            encoder.zero_grad()
            classifier.zero_grad()

            text_tensor = batch.Text[0]
            label_tensor = batch.Label
            out = encoder(text_tensor)
            score, attn = classifier(out)
            batch_loss = loss_function(score, label_tensor)
            batch_loss.backward()
            optimizer.step()
            all_loss += batch_loss.item()
        print("epoch", epoch, "\t", "loss", all_loss)



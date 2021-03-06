import collections
import os
import pandas as pd
from glob import glob
import linecache

import MeCab
import re
import torch

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import LSTMClassifier


tagger = MeCab.Tagger("-Owakati")

def make_wakati(sentence):
    # wakati by mecab
    sentence = tagger.parse(sentence)
    # 半角全角英数字除去
    sentence = re.sub(r'[0-9０-９a-zA-Zａ-ｚＡ-Ｚ]+', " ", sentence)
    # 記号もろもろ除去
    sentence = re.sub(r'[\．_－―─！＠＃＄％＾＆\-‐|\\＊\“（）＿■×+α※÷⇒—●★☆〇◎◆▼◇△□(：〜～＋=)／*&^%$#@!~`){}［］…\[\]\"\'\”\’:;<>?＜＞〔〕〈〉？、。・,\./『』【】「」→←○《》≪≫\n\u3000]+', "", sentence)
    # スペースで区切って形態素の配列へ
    wakati = sentence.split(" ")
    # 空の要素は削除
    wakati = list(filter(("").__ne__, wakati))
    return wakati


def get_title():
    categories = [name for name in os.listdir("text") if os.path.isdir("text/" + name)]
    #print(categories)

    datasets = pd.DataFrame(columns=["title", "category"])
    categories_test = categories[:3]

    for cat in categories_test:
        path = "text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            #print(title)
            s = pd.Series([title, cat], index=datasets.columns)
            datasets = datasets.append(s, ignore_index=True)

    datasets = datasets.sample(frac=1).reset_index(drop=True)

    return datasets

def get_title_as_list():
    categories = [name for name in os.listdir("text") if os.path.isdir("text/" + name)]
    #print(categories)

    categories_test = categories[:3]

    titles = []
    categories = []

    for cat in categories_test:
        path = "text/" + cat + "/*.txt"
        files = glob(path)
        for text_name in files:
            title = linecache.getline(text_name, 3)
            #print(title)
            titles.append(title)
            categories.append(categories)


    return titles, categories 



def sentence2index(sentence, word2index):
    wakati = make_wakati(sentence)
    return torch.tensor([word2index[w] for w in wakati], dtype=torch.long)

def category2tensor(cat, category2index):
    return torch.tensor([category2index[cat]], dtype=torch.long)


# function to make batch
def train2batch(title, category, batch_size=100):
    title_batch = []
    category_batch = []
    title_shuffle, category_shuffle = shuffle(title, category)
    for i in range(0, len(title), batch_size):
        title_batch.append(title_shuffle[i:i+batch_size])
        category_batch.append(category_shuffle[i:i+batch_size])
    return title_batch, category_batch


class MyDataManager(Dataset):
    def __init__(self, ds, ls):
        super(MyDataManager, self).__init__()
        self.ds = ds
        self.ls = ls

    def __getitem__(self, idx):
        return self.ds[idx], self.ls[idx]


    def __len__(self):
        return len(self.ds)




if __name__ == "__main__":

    #test = "僕はドラえもんです。"
    #print(make_wakati(test))
    datasets = get_title()

    word2index = {}
    word2index.update({"<pad>":0})

    for title in datasets["title"]:
        wakati = make_wakati(title)
        for word in wakati:
            if word in word2index: continue
            word2index[word] = len(word2index)

    print("vocab size: ", len(word2index))

    test = "例のあのメニューも！ニコニコ超会議のフードコートメニュー14種類紹介（前半）"
    print(sentence2index(test, word2index))


    embedding_dim = 200
    hidden_dim = 128
    vocab_size = len(word2index)


    inputs = sentence2index(test, word2index)

    embeds = nn.Embedding(vocab_size, embedding_dim)
    sentence_matrix = embeds(inputs)


    sentence_matrix = sentence_matrix.view(sentence_matrix.size()[0], 1, sentence_matrix.size()[1])

    lstm = nn.LSTM(embedding_dim, hidden_dim)

    out1, (h, c) = lstm(sentence_matrix)

    print(sentence_matrix.size())
    print(out1.size())
    print(h.size())
    print(c.size())

    categories = [name for name in os.listdir("text") if os.path.isdir("text/" + name)]
    category2index = {}
    for cat in categories:
        if cat in category2index: continue
        category2index[cat] = len(category2index)

    print(category2index)

    print(category2tensor("it-life-hack", category2index))

    index_datasets_title_tmp = []
    index_datasets_category = []

    # pad to adjust the length to the longest one
    max_len = 0
    for title, category in zip(datasets["title"], datasets["category"]):
        index_title = sentence2index(title, word2index)
        index_category = category2tensor(category, category2index)
        index_datasets_title_tmp.append(index_title)
        index_datasets_category.append(index_category)
        if max_len < len(index_title):
            max_len = len(index_title)

    # if the length is shorter than max_len, pad in front
    index_datasets_title = []
    for title in index_datasets_title_tmp:
        for i in range(max_len - len(title)):
            #print("****")
            #print(type(title))
            #print(title.shape)
            title = torch.cat([torch.FloatTensor([0]), title])
            #title.insert(0, 0)
        index_datasets_title.append(title)

    train_x, test_x, train_y, test_y = train_test_split(index_datasets_title, index_datasets_category, train_size=0.7)



    # prepare the data
    traindata, testdata = train_test_split(datasets, train_size=0.7)

    print("traindata")
    print(len(traindata))
    print(type(traindata))
    print(traindata.head())
    print(traindata.iloc[0])

    #train_datasets = MyDataManager(traindata["title"], traindata["category"])
    #test_datasets = MyDataManager(testdata["title"], testdata["category"])

    #train_kwargs = {"shuffle":True, "batch_size": 32}
    #test_kwargs = {"batch_size": 32}

    #train_dataloader = DataLoader(train_datasets, **train_kwargs)
    #test_dataloader = DataLoader(test_datasets, **test_kwargs)

    # training 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tag_size = len(categories)
    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, tag_size).to(device)
    epochs = 5
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    loss_function = nn.NLLLoss()


    #for epoch in range(epochs):
    #    for i, (data, target) in enumerate(train_dataloader):
    #        batch_loss = 0
    #        model.zero_grad()
    #        out = model(data)
    #        batch_loss = loss_function(out, category_tensor)
    #        batch_loss.backward()
    #        optimizer.step()
    #        all_loss += batch_loss.item()

    #    print("epoch", epoch, "\t", "loss", all_loss)
    #    if all_loss < 0.1: break
    #print("done with dataloder.")


    for epoch in range(epochs):
        all_loss = 0
        title_batch, category_batch = train2batch(train_x, train_y)
        for i in range(len(title_batch)):
            batch_loss = 0

            model.zero_grad()
            # pass device=GPU to inference the tensor with GPU
            title_tensor = torch.zeros(len(title_batch[i]), title_batch[i][0].size()[0]).long()
            for j in range(len(title_tensor)):
                title_tensor[j, :] = title_batch[i][j]
            #title_tensor = torch.tensor(title_batch[i], device=device)
            # category_tensor.size() = (batch_size * 1) , so squeeze()
            category_tensor = torch.tensor(category_batch[i], device=device).squeeze()

            out = model(title_tensor)

            batch_loss = loss_function(out, category_tensor)
            batch_loss.backward()

            optimizer.step()

            all_loss += batch_loss.item()

        print("epoch", epoch, "\t", "loss", all_loss)

        if all_loss < 0.1: break
    print("done.")


    test_num = len(test_x)

    a = 0

    with torch.no_grad():
        title_batch, category_batch = train2batch(test_x, test_y)

        for i in range(len(title_batch)):
            title_tensor = torch.zeros(len(title_batch[i]), title_batch[i][0].size()[0]).long()
            for j in range(len(title_batch[i])):
                title_tensor[j, :] = title_batch[i][j]
            #title_tensor = torch.tensor(title_batch[i], device=device)
            category_tensor = torch.tensor(category_batch[i], device=device)

            out = model(title_tensor)

            _, predicts = torch.max(out, 1)
            for j, ans in enumerate(category_tensor):
                if predicts[j].item() == ans.item():
                    a += 1
    print("predict : ", a / test_num)





#    losses = []
#
#    for epoch in range(epochs):
#        all_loss = 0
#        for i, (title, cat) in enumerate(zip(traindata["title"], traindata["category"])):
#            model.zero_grad()
#            inputs = sentence2index(title, word2index)
#
#            out = model(inputs)
#
#            answer = category2tensor(cat, category2index)
#
#            loss = loss_function(out, answer)
#
#            loss.backward()
#
#            optimizer.step()
#
#            all_loss += loss.item()
#
#        losses.append(all_loss)
#        print("epoch", epoch, "\t", "loss", all_loss)
#
#    print("done.")
#
#    plt.plot(losses)
#
#
#    # validataion with testdata
#    test_num = len(testdata)
#
#    a = 0
#
#    with torch.no_grad():
#        for i, (title, category) in enumerate(zip(testdata["title"], testdata["category"])):
#
#            inputs = sentence2index(title, word2index)
#            out = model(inputs)
#
#            _, predict = torch.max(out, 1)
#
#            answer = category2tensor(category, category2index)
#
#            if predict == answer:
#                a += 1
#
#    print("predict: {:.3f}".format(a/test_num))
#
#
#    # validataion with train data (to check if the model is over training)
#    traindata_num = len(traindata)
#    a = 0
#    with torch.no_grad():
#        for i, (title, category) in enumerate(zip(traindata["title"], traindata["category"])):
#            inputs = sentence2index(title, word2index)
#            out = model(inputs)
#
#            _, predict = torch.max(out, 1)
#            answer = category2tensor(category, category2index)
#            if predict == answer:
#                a += 1
#
#    print("predict : {:.3f}".format( a/traindata_num))
#
#
#    index2category = {}
#
#    for cat, idx in category2index.items():
#        index2category[idx] = cat
#
#    # answer -> correct label, predict -> lstm prediction, exact -> 0 if correct, X if incorrect
#    predict_df = pd.DataFrame(columns=["answer", "predict", "exact"])
#
#    with torch.no_grad():
#        for title, category in zip(testdata["title"], testdata["category"]):
#            out = model(sentence2index(title, word2index))
#            _, predict = torch.max(out, 1)
#            answer = category2tensor(category, category2index)
#            exact = "O" if predict.item() == answer.item() else "X"
#            s = pd.Series([answer.item(), predict.item(), exact], index=predict_df.columns)
#            predict_df = predict_df.append(s, ignore_index=True)
#
#
#    fscore_df = pd.DataFrame(columns=["category", "all", "precision", "recall", "fscore"])
#
#    prediction_count = collections.Counter(predict_df["predict"])
#
#    answer_count = collections.Counter(predict_df["answer"])
#
#    for i in range(len(category2index)):
#        all_count = answer_count[i]
#        precision = len(predict_df.query('predict == ' + str(i) + ' and exact == "O"'))/prediction_count[i]
#        recall = len(predict_df.query('answer == ' + str(i) + ' and exact == "O"')) / all_count
#        fscore = 2 * precision * recall / (precision + recall)
#        s = pd.Series([index2category[i], all_count, round(precision, 2), round(recall, 2), round(fscore, 2)], index=fscore_df.columns)
#        fscore_df = fscore_df.append(s, ignore_index=True)
#
#    print(fscore_df)



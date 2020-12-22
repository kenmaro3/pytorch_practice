import itertools 

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR


import os
from os.path import join as osp
from natsort import natsorted
from sklearn.model_selection import train_test_split
import numpy as np

from PIL import Image

import itertools

from model import *

import matplotlib.pyplot as plt

from transfer_alex import MyModel
from food_like_model import TestModel

class MyDataManager(Dataset):
    def __init__(self, ds, ls):
        super(MyDataManager, self).__init__()
        self.ds = ds
        self.ls = ls

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx):
        img = Image.open(self.ds[idx])
        img = self.transform(img)

        l = self.ls[idx]

        return img, l

    def __len__(self):
        return len(self.ds)


def visualize(ds, idx):
    img_data = Image.open(ds[idx])
    #target_data = Iimage.open(ls[idx])

    #fig, ax = plt.subplot(1,2)
    plt.imshow(img_data)

    plt.show()



def train(model, train_loader,optimizer, criterion, device, log_interval, epoch_idx):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        if i % log_interval == 0:
            print(
                    "train epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(
                        epoch_idx,
                        i * len(data),
                        len(train_loader.dataset),
                        100.0 * i /len(train_loader),
                        loss.item()
                        )
            )

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)

            prediction = output.argmax(dim=1, keepdim=True)

            correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset)
                )
        )










if __name__ == "__main__":
    print("test")


    train_path = "./Images"

    target_list = os.listdir(train_path)
    print(len(target_list))
    print(target_list)
   
    #target_list = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]
    
    test_size=100
    tmp1 = []
    for i, target in enumerate(target_list[:test_size]):
      file_list = os.listdir(osp(train_path, target))
      file_list2 = [osp(train_path, target, file_) for file_ in file_list]

      tmp1.append(file_list2)

    print(len(tmp1))
    print(len(tmp1[0]))

    #target_files = [osp(train_path, target, os.listdir(osp(train_path, target))) for target in target_list]
    print(tmp1[0][0])

    tmp2 = []
    for i in range(len(tmp1)):
      tmp3 = []
      for j in range(len(tmp1[i])):
        tmp3.append(i)
      tmp2.append(tmp3)


    print(len(tmp2))
    print(len(tmp2[0]))
    
    ds = list(itertools.chain.from_iterable(tmp1))
    ls = list(itertools.chain.from_iterable(tmp2))
    #print(ds)
    #print(ls)
    
    ls_np = np.zeros(len(ls), dtype=np.int64)
    for i in range(len(ls)):
      ls_np[i] = ls[i]

    print(ls_np)

    ##ls = torch.LongTensor(ls)


    train_kwargs = dict(
        batch_size=32,
        shuffle=True,
    )

    test_kwargs = dict(
        batch_size=12
    )

    #print("here000")
    #print(type(ds))
    #print(len(ds))
    #print(type(ls))
    #print(len(ls))


    train_x, test_x, train_y, test_y = train_test_split(ds, ls, test_size=0.2)
    #
    #print("here1111")

    train_dataset = MyDataManager(train_x, train_y)
    test_dataset = MyDataManager(test_x, test_y)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)


    device = torch.device('cpu')

    model = TestModel(len(tmp1)).to(device)

    #test = torch.randn(1, 3, 224, 224)

    #output = model(test)
    ##model = MyNet().to(device)
    #model = MyModel(9216, 2).to(device)

    #model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    #for param in model.parameters():
    #  param.requires_grad = False

    #model.classifier = nn.Sequential(
    #  nn.Linear(9216, 2048),
    #  nn.ReLU(inplace=True),
    #  nn.Dropout(0.4),
    #  nn.Linear(2048, 2),
    #  nn.LogSoftmax(dim=1)
    #)

    optimizer = optim.Adam(model.parameters())


    criterion = nn.CrossEntropyLoss()
    ##criterion = nn.NLLLoss()

    ##test = torch.randn(1, 3, 224, 224)

    ##target = torch.LongTensor([1]) 

    ##output = model(test)

    ##print(output.size())

    ##loss = criterion(output, target)
    ##print(loss.item())

    epochs = 100
    log_interval = 10
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)
        if e % 1 == 0:
          torch.save(model, f'test_model_epoch_{e}.pth')





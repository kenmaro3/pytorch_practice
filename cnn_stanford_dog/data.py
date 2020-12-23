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

class MyDataManager(Dataset):
    def __init__(self, ds, ls):
        super(MyDataManager, self).__init__()
        self.ds = ds
        self.ls = ls

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
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



def main():

    print("test")
    target_list = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]

    train_path = "./Images"

    target_files = [os.listdir(osp(train_path, target)) for target in target_list]

    ds = []
    target_full_path1 = [osp(train_path, target_list[0], file) for file in target_files[0]]
    target_full_path2 = [osp(train_path, target_list[1], file) for file in target_files[1]]

    ds = target_full_path1 + target_full_path2

    print(len(ds))
    print(ds[0])

    #visualize(ds, 0)

    total_length = len(target_files[0]) + len(target_files[1])

    ls = np.zeros(total_length, dtype=np.int64)
    for i in range(len(target_files[0])):
        ls[i] = 1

    #ls = torch.zeros(total_length, dtype=torch.long)
    #for i in range(len(target_files[0])):
    #    ls[i] = 1


    #ls = torch.LongTensor(ls)


    train_kwargs = dict(
        batch_size=32,
        shuffle=True,
    )

    test_kwargs = dict(
        batch_size=12
    )


    train_x, test_x, train_y, test_y = train_test_split(ds, ls, test_size=0.2)
    

    train_dataset = MyDataManager(train_x, train_y)
    test_dataset = MyDataManager(test_x, test_y)

    train_loader = DataLoader(train_dataset, **train_kwargs)
    test_loader = DataLoader(test_dataset, **test_kwargs)


    device = torch.device('cpu')
    #model = MyNet().to(device)
    #model = MyModel(9216, 2).to(device)

    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    for param in model.parameters():
      param.requires_grad = False

    model.classifier = nn.Sequential(
      nn.Linear(9216, 2048),
      nn.ReLU(inplace=True),
      nn.Dropout(0.4),
      nn.Linear(2048, 2),
      nn.LogSoftmax(dim=1)
    )

    optimizer = optim.Adam(model.parameters())


    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    #test = torch.randn(1, 3, 224, 224)

    #target = torch.LongTensor([1]) 

    #output = model(test)

    #print(output.size())

    #loss = criterion(output, target)
    #print(loss.item())

    epochs = 100
    log_interval = 10
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)


if __name__ == "__main__":
  main()

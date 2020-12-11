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
    target_list = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]

    train_path = "/Users/kmihara/Downloads/Images"

    target_files = [os.listdir(osp(train_path, target)) for target in target_list]

    ds = []
    target_full_path1 = [osp(train_path, target_list[0], file) for file in target_files[0]]
    target_full_path2 = [osp(train_path, target_list[1], file) for file in target_files[1]]

    ds = target_full_path1 + target_full_path2

    print(len(ds))
    print(ds[0])

    #visualize(ds, 0)

    total_length = len(target_files[0]) + len(target_files[1])

    ls = torch.zeros(total_length, dtype=torch.long)
    for i in range(len(target_files[0])):
        ls[i] = 1


    ls = torch.LongTensor(ls)


    train_kwargs = dict(
        batch_size=12,
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
    model = MyNet().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.1)


    criterion = nn.CrossEntropyLoss()

    epochs = 5
    log_interval = 10
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)





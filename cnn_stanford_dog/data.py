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
            #transforms.Resize((128, 128)),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([.5,.5,.5], [.5,.5,.5]),
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




def dispense_dataloader(num_labels):

    train_path = "./Images"

    target_list = os.listdir(train_path)

    #target_list = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]
    remove_list = ["n02105855-Shetland_sheepdog"] # this dataset is corrupted...

    tmp1 = []
    for i, target in enumerate(target_list[:num_labels]):
      if target in remove_list:
        continue
      file_list = os.listdir(osp(train_path, target))
      file_list2 = [osp(train_path, target, file_) for file_ in file_list]

      tmp1.append(file_list2)


    tmp2 = []
    for i in range(len(tmp1)):
      tmp3 = []
      for j in range(len(tmp1[i])):
        tmp3.append(i)
      tmp2.append(tmp3)


    ds = list(itertools.chain.from_iterable(tmp1))
    ls = list(itertools.chain.from_iterable(tmp2))

    ls_np = np.zeros(len(ls), dtype=np.int64)
    for i in range(len(ls)):
      ls_np[i] = ls[i]

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

    return train_loader, test_loader


def dispense_dataloader_specific(num_labels):

    train_path = "./Images"

    target_list = os.listdir(train_path)

    target_list_specific = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]
    remove_list = ["n02105855-Shetland_sheepdog"] # this dataset is corrupted...

    tmp1 = []
    for i, target in enumerate(target_list[:num_labels]):
      #print(f'target >>> {target}')
      if target in remove_list:
        continue
      if target in target_list_specific:
        print(f'target i >>> {i}')
      file_list = os.listdir(osp(train_path, target))
      file_list2 = []
      for j in range(len(file_list)):
          if target in target_list_specific:
              if j == 100:
                  break
          file_list2.append(osp(train_path, target, file_list[j]))
        #file_list2 = [osp(train_path, target, file_) for file_ in file_list]

      tmp1.append(file_list2)


    tmp2 = []
    for i in range(len(tmp1)):
      tmp3 = []
      for j in range(len(tmp1[i])):
        tmp3.append(i)
      tmp2.append(tmp3)


    ds = list(itertools.chain.from_iterable(tmp1))
    ls = list(itertools.chain.from_iterable(tmp2))

    ls_np = np.zeros(len(ls), dtype=np.int64)
    for i in range(len(ls)):
      ls_np[i] = ls[i]

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

    return train_loader, test_loader


def dispense_dataloader_special():

    train_path = "./data_for_sikibetu"

    target_list = os.listdir(train_path)

    #target_list = ["n02094114-Norfolk_terrier", "n02094258-Norwich_terrier"]
    #remove_list = ["n02105855-Shetland_sheepdog"] # this dataset is corrupted...

    tmp1 = []
    for i, target in enumerate(target_list):
      file_list = os.listdir(osp(train_path, target))
      file_list2 = [osp(train_path, target, file_) for file_ in file_list]

      tmp1.append(file_list2)


    tmp2 = []
    for i in range(len(tmp1)):
      tmp3 = []
      for j in range(len(tmp1[i])):
        tmp3.append(i)
      tmp2.append(tmp3)


    ds = list(itertools.chain.from_iterable(tmp1))
    ls = list(itertools.chain.from_iterable(tmp2))

    ls_np = np.zeros(len(ls), dtype=np.int64)
    for i in range(len(ls)):
      ls_np[i] = ls[i]

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

    return train_loader, test_loader


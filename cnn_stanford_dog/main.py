import shutil
import os
from os.path import join as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import visualize, dispense_dataloader, dispense_dataloader_specific, dispense_dataloader_special
from train_test import train, test
from food_like_model import TestModel


def train_mynet_with_whole_dataset(cp_folder):
    if os.path.exists(cp_folder):
        shutil.rmtree(cp_folder)
    os.mkdir(cp_folder)

    num_labels=120

    train_loader, test_loader = dispense_dataloader_specific(num_labels=num_labels)


    machine_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'machiine_type >>> {machine_type}')
    device = torch.device(machine_type)

    model = TestModel(num_labels).to(device)
    print(f'loaded model >>> {model}')
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    print(f'optimizer >>> {optimizer}')


    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    print(f'criterion >>> {criterion}')

    epochs = 200
    log_interval = 10
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    print(f'epochs >>> {epochs}')
    print(f'log_interval >>> {log_interval}')
    print(f'scheduler >>> {scheduler}')



    print(f'\ntrain start...')
    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)
        if e % 10 == 0:
          torch.save(model, osp(cp_folder, f'test_model_epoch_{e}.pth'))
          print(f'{e} th iteration done. now we have lr as >>> {scheduler.get_last_lr()}')



def transfer_alex(cp_folder):
    if os.path.exists(cp_folder):
        shutil.rmtree(cp_folder)
    os.mkdir(cp_folder)

    num_labels=120

    train_loader, test_loader = dispense_dataloader_special()


    machine_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'machiine_type >>> {machine_type}')
    device = torch.device(machine_type)

    #model = TestModel(num_labels).to(device)
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
        model.classifier = nn.Sequential(
        nn.Linear(9216, 2048),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(2048, 4),
        #nn.LogSoftmax(dim=1)
      )

    print(f'loaded model >>> {model}')
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    print(f'optimizer >>> {optimizer}')


    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    print(f'criterion >>> {criterion}')

    epochs = 200
    log_interval = 10
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    #scheduler = CosineAnnealingLR(optimizer, epochs)

    print(f'epochs >>> {epochs}')
    print(f'log_interval >>> {log_interval}')
    print(f'scheduler >>> {scheduler}')



    print(f'\ntrain start...')
    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)
        if e % 10 == 0:
          torch.save(model, osp(cp_folder, f'test_model_epoch_{e}.pth'))
          print(f'{e} th iteration done. now we have lr as >>> {scheduler.get_last_lr()}')






if __name__ == "__main__":
    cp_folder = "cp_alex_transfer"
    transfer_alex(cp_folder)

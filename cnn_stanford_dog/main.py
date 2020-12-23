import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR

from whole_dog import visualize, dispense_dataloader, dispense_dataloader_specific
from train_test import train, test
from food_like_model import TestModel


if __name__ == "__main__":

    print("test")

    num_labels=100

    train_loader, test_loader = dispense_dataloader_specific(num_labels=num_labels)


    machine_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(machine_type)

    print("model")
    model = TestModel(num_labels).to(device)
    optimizer = optim.Adam(model.parameters())


    criterion = nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()

    epochs = 100
    log_interval = 10
    #scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    scheduler = CosineAnnealingLR(optimizer, epochs)

    for e in range(epochs):
        train(model, train_loader, optimizer, criterion, device, log_interval, e)
        test(model, test_loader, criterion, device)
        if e % 10 == 0:
          torch.save(model, f'test_model_epoch_{e}.pth')
          print(f'{e} th iteration done. now we have lr as >>> {scheduler.get_lr()}')





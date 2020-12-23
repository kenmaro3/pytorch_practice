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
          torch.save(model, f'test_model_epoch_{e}.pth')
          print(f'{e} th iteration done. now we have lr as >>> {scheduler.get_last_lr()}')





from natsort import natsorted
import cv2
from os.path import join as osp
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from sklearn.model_selection import train_test_split


from unet import UNet


import shutil


class MyDataManager(Dataset):
  def __init__(self, ds, ls):
    super(MyDataManager, self).__init__()
    self.ds = ds
    self.ls = ls
    self.scale = 0.33

    self.transform = transforms.Compose([
      transforms.Resize((480, 640)),
      transforms.ToTensor()
    ])

  def __getitem__(self, idx):
    data_img = Image.open(self.ds[idx])
    data_img = self.transform(data_img)

    target_img = Image.open(self.ls[idx])
    target_img = self.transform(target_img)

    return data_img, target_img

  def __len__(self):
    return len(self.ds)



def visualize(ds, ls, idx):
  img_data = Image.open(ds[idx])
  target_data = Image.open(ls[idx])
  x = transforms.ToTensor()(img_data)
  
  fig, ax = plt.subplots(1, 2)
  
  ax[0].imshow(img_data)
  ax[1].imshow(target_data)

  #plt.imshow(img)
  plt.show()
  
  
def train(model, train_loader, device, epochs, log_interval, criterion, optimizer):
  model.train()
  for e in range(epochs):
    for i, (data, target) in enumerate(train_loader):
      data= data.to(device=device, dtype=torch.float32)
      target = target.to(device=device, dtype=torch.float32)
      optimizer.zero_grad()
      output = model(data)
      
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      if i % log_interval:
        print(
          "train epoch: {} [{}/{} ({:.0f}%] \tLoss: {:.6f}".format(
           e,
           i * len(data),
           len(train_loader.dataset),
           100.0*i/len(train_loader),
           loss.item()
          )
        )


def test(model, test_loader, device, criterion):
  model.eval()
  test_loss = 0
  correct = 0

  with torch.no_grad():
    for data, target in test_loader:
      data = data.to(device, dtype=torch.float32)
      target = target.to(device, dtype=torch.float32)
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
        100.0 * correct /len(test_loader.datset)
      )
    )


def save_model(model, dir_checkpoint, epoch_num):
  torch.save(model.state_dict(), osp(dir_checkpoint, f'CP_epoch{epoch_num}.pth'))
  

def create_dataset(folder_data, folder_target):

  data_list= os.listdir(folder_data)
  target_list = os.listdir(folder_target)

  data_list = natsorted(data_list)
  target_list = natsorted(target_list)
  
  ds = [osp(folder_data, img) for img in data_list]
  ls = [osp(folder_target, img) for img in target_list]

  train_x, test_x, train_y, test_y = train_test_split(ds, ls, test_size=0.2)

  return train_x, test_x, train_y, test_y


def create_data_loader(train_x, test_x, train_y, test_y):

  train_dataset = MyDataManager(train_x, train_y)
  test_dataset = MyDataManager(test_x, test_y)

  train_kwargs = dict(
    batch_size=4,
    shuffle=True
  )

  test_kwargs = dict(
    batch_size=4,
  )

  
  train_loader = DataLoader(train_dataset, **train_kwargs)
  test_loader = DataLoader(test_dataset, **test_kwargs)

  return train_loader, test_loader



if __name__ == "__main__":
  folder_data = "/home/kmihara/Downloads/carvana/train"
  folder_target = "/home/kmihara/Downloads/carvana/train_masks"
  
  train_x, test_x, train_y, test_y  = create_dataset(folder_data=folder_data, folder_target=folder_target)

  print(f'created dataset')

  train_loader, test_loader = create_data_loader(train_x, test_x, train_y, test_y)

  print(f'created data loader')

  dir_checkpoint = "checkpoint_test"
  if os.path.exists(dir_checkpoint):
    shutil.rmtree(dir_checkpoint)
    os.mkdir(dir_checkpoint)
  else:
    os.mkdir(dir_checkpoint)

  epochs = 5
  device = torch.device('cpu')
  batch_size = 12
  lr = 0.1
  log_interval = 5

  
  criterion = nn.BCEWithLogitsLoss()

  model = UNet(n_channels=3, n_classes=1)
  optimizer = optim.SGD(model.parameters(), lr=lr)
  #optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=1e-8, modentum=0.9)
  scheduler = StepLR(optimizer, step_size=1, gamma=0.1) 


  scheduler = StepLR(optimizer, step_size=1, gamma = 0.1)

  print(f'start training')
  for e in range(epochs):
    train(model, train_loader, device, epochs, log_interval, criterion, optimizer)
    test(model, test_loader, device, criterion) 
    scheduler.step()
    print(f'save model for checkpoint')
    save_model(model, dir_checkpoint, e)
  
  print(f'finished training')



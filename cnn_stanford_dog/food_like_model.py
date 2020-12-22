import torch
import torch.nn as nn
import torch.nn.functional as F




class TestModel(nn.Module):
  def __init__(self, n_classes):
    super(TestModel, self).__init__()

    self.conv1 = nn.Conv2d(3,12, kernel_size=7, stride=4, padding=1)
    self.bn1 = nn.BatchNorm2d(12)
    self.conv2 = nn.Conv2d(12,12,  kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(12)
    self.conv3 = nn.Conv2d(12,46, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(46)
    self.mp1 = nn.MaxPool2d(kernel_size=2)
    self.conv4 = nn.Conv2d(46,64,  kernel_size=3, stride=1, padding=1)
    self.bn4 = nn.BatchNorm2d(64)
    self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.bn5 = nn.BatchNorm2d(64)
    self.mp2 = nn.MaxPool2d(kernel_size=2)
    self.conv6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn6 = nn.BatchNorm2d(128)
    self.conv7 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.bn7 = nn.BatchNorm2d(128)
    self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.bn8 = nn.BatchNorm2d(256)

    self.gap = nn.AdaptiveAvgPool2d(1)

    self.fc1 = nn.Linear(256, 256)
    self.fc2 = nn.Linear(256, n_classes) 

    #self.sigmoid = nn.Sigmoid()

  
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = self.mp1(x)
    x = F.relu(self.bn4(self.conv4(x)))
    x = F.relu(self.bn5(self.conv5(x)))
    x = self.mp2(x)
    x = F.relu(self.bn6(self.conv6(x)))
    x = F.relu(self.bn7(self.conv7(x)))
    x = F.relu(self.bn8(self.conv8(x)))

    #print(f'conv8 >>> {x.shape}')

    x = self.gap(x)
    #print(f'gap >>> {x.shape}')
    x = torch.squeeze(x) 
    #print(f'x >>> {x.shape}')

    x = F.relu(self.fc1(x))
    x = self.fc2(x)

    #return self.sigmoid(x)
    return x

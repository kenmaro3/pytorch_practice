import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR



class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(5)
        self.bn2 = nn.BatchNorm2d(5)
        self.mp1 = nn.MaxPool2d(2,2)
        self.mp2 = nn.MaxPool2d(2,2)

        self.fc1 = nn.Linear(5120, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = self.bn2(x)

        x = nn.Flatten()(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2304, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )


    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        x = self.classifier(x)
        return x



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3,20, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5, stride=2)
        self.mp1 = nn.MaxPool2d(2,2)
        self.mp2 = nn.MaxPool2d(2,2)

        self.bn1 = nn.BatchNorm2d(20)
        self.bn2 = nn.BatchNorm2d(20)

        self.fc1 = nn.Linear(980, 128)
        self.fc2 = nn.Linear(128, 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = self.bn2(x)

        x = nn.Flatten()(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x



class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))



class Block(nn.Module):
    def __init__(self, c_in, c_out):
        super(Block, self).__init__()
        channel = c_out // 4

        # 1x1 conv
        self.conv1 = nn.Conv2d(c_in, channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu1 = nn.ReLU()

        # 3x3 conv
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channel)
        self.relu2 = nn.ReLU()

        # 1x1 conv
        self.conv3 = nn.Conv2d(channel, c_out, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(c_out)

        # adjust channel for skip connection
        self.shortcut = self.shortcut(c_in, c_out)

        self.relu3 = nn.ReLU()

    
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        y = self.relu3(h + shortcut)
        return y

    def _shortcut(self, c_in, c_out):
        if c_in != c_out:
            return self._projection(c_in, c_out)
        else:
            return lambda x: x

    def _projection(self, c_in, c_out):
        return nn.Conv2d(c_in, c_out,
                kernel_size=(1,1),
                padding=0)



class ResNet50(nn.Module):
    def __init__(self, output_dim):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,
                            kernel_size=7,
                            stride=2,
                            padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Block1
        self.block0 = self._building_block(256, c_in=64)
        self.block1 = nn.ModuleList([
            self._building_block(256) for _ in range(2)
            ])

        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=2)

        # Block 2
        self.block2 = nn.ModuleList([
            self._building_block(512) for _ in range(4)
            ])

        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)

        # Block 3
        self.block3 = nn.ModuleList([
            self._building_block(1024) for _ in range(6)
            ])

        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2)

        # Block 4
        self.block4 = nn.ModuleList([
            self._building_block(2048) for _ in range(3)
            ])

        self.avg_pool = GlobalAvgPool2d()
        self.fc = nn.Linear(2048, 1000)
        self.out = nn.Linear(1000, output_dim)



    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.pool1(h)
        h = self.block0(h)

        for block in self.block1:
            h = block(h)

        h = self.conv2(h)

        for block in self.block2:
            h = block(h)

        h = self.conv3(h)

        for block in self.block3:
            h = block(h)

        h = self.conv4(h)

        for block in self.block4:
            h = block(h)

        h = self.avg_pool(h)
        h = self.fc(h)
        h = F.relu(h)
        h = self.out(h)
        y = torch.log_softmax(h, dim=-1)

        return y


    def _building_block(self, c_out, c_in=None):
        if c_in is None:
            c_in = c_out
        return Block(c_in, c_out)


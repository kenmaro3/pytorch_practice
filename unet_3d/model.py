import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
  # (convolution => bn => relu)*2
  def __init__(self, inc, outc, midc=None):
    super(DoubleConv, self).__init__()
    if not midc:
      midc = outc
    self.double_conv = nn.Sequential(
      nn.Conv3d(inc, midc, kernel_size=(3,3,3), padding=(1,1,1)),
      nn.BatchNorm3d(midc),
      nn.ReLU(inplace=True),
      nn.Conv3d(midc, outc, kernel_size=(3,3,3), padding=(1,1,1)),
      nn.BatchNorm3d(outc),
      nn.ReLU(inplace=True)
    )


  def forward(self, x):
    return self.double_conv(x)

  

class Down(nn.Module):
  # (maxpool for (h, w) not d => DoubleConv)
  def __init__(self, inc, outc):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1,2,2)),
      DoubleConv(inc, outc)
    )

  def forward(self, x):
    # x.size() = (batch, inc, d, h, w) => (batch, outc, d, h//2, w//2) 
    return self.maxpool_conv(x)


class Up(nn.Module):
  # ( ConvTranspose3d for (h, w) dimension, not d => DoubleConv)
  def __init__(self, inc, outc):
    super(Up, self).__init__()

    # since inc is after torch.cat of x1 and x2, self.up will halve the channel
    self.up = nn.ConvTranspose3d(inc, inc//2, kernel_size=(1,2,2), stride=(1,2,2))
    self.conv = DoubleConv(inc, outc)


  def forward(self, x1, x2):
    # x1.size() = (batch, inc//2, d, h, w]
    # x2.size() = (batch, inc//2, d, h, w]
    x1 = self.up(x1)

    # input is CDHW, [bn, c, d, h, w]

    diffD = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[4] - x1.size()[4]

    #print("yooyoyoyoyo"*10)
    #print(x1.size())
    #print(x2.size())

    #print("diffD")
    #print(diffD)
    #print("diffX")
    #print(diffX)
    #print("diffY")
    #print(diffY)

    padding_list = []

    assert not diffD < 0
    assert not diffX < 0
    assert not diffY < 0

    if diffY == 1:
      padding_list.append(1)
      padding_list.append(0)
    else:
      padding_list.append(diffY//2)
      padding_list.append(diffY-diffY//2)

    if diffX == 1:
      padding_list.append(1)
      padding_list.append(0)
    else:
      padding_list.append(diffX//2)
      padding_list.append(diffX-diffX//2)


    if diffD == 1:
      padding_list.append(1)
      padding_list.append(0)
    else:
      padding_list.append(diffD//2)
      padding_list.append(diffD-diffD//2)


    #print("****"*20)
    #print(padding_list)

    

    x1 = F.pad(x1, padding_list)
    #x1 = F.pad(x1, [diffD//2, diffD-diffD//2, diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])

    #print("dude"*10)
    #print(x1.size())
    #print(x2.size())

    # x.size() = (batch, cin, d, h, w)
    x = torch.cat([x2, x1], dim=1)
    
    # self.conv(x).size() = (batch, cin//2, d, 2*h, 2*w)
    return self.conv(x)


class OutConv(nn.Module):
  # last conv which maps to the same dimension of target
  # use kernel_size=1, padding=0, stride=1
  def __init__(self, inc, outc):
    super(OutConv, self).__init__()
    self.conv = nn.Conv3d(inc, outc, kernel_size=1)
  
  def forward(self, x):
    # x.size() = (batch, inc, d, h, w)
    # self.conv(x).size() = (batch, outc, d, h, w)
    return self.conv(x)
    
    

class UNet3D(nn.Module):
  '''
    n_channels = c of input = 3
    n_classes = c of target = 1
  '''
  def __init__(self, n_channels, n_classes):
    super(UNet3D, self).__init__() 
    self.n_channels = n_channels
    self.n_classes = n_classes
    
    self.inc = DoubleConv(n_channels, 64)

    self.down1 = Down(64, 128)
    self.down2 = Down(128, 256)
    self.down3 = Down(256, 512)
    self.down4 = Down(512, 1024)
    
    self.up1 = Up(1024, 512)
    self.up2 = Up(512, 256)
    self.up3 = Up(256, 128)
    self.up4 = Up(128, 64)

    self.outc = OutConv(64, n_classes)


  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.down1(x1)
    x3 = self.down2(x2)
    x4 = self.down3(x3)
    x5 = self.down4(x4)

    x = self.up1(x5, x4)
    x = self.up2(x, x3)
    x = self.up3(x, x2)
    x = self.up4(x, x1)

    logits = self.outc(x)

    return logits


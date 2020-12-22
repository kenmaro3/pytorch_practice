import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
  # (conv => bn => relu)*2
  def __init__(self, inc, outc, midc=None):
    super(DoubleConv, self).__init__()
    if not midc:
      midc = outc
    self.double_conv = nn.Sequential(
      nn.Conv3d(inc, midc, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)), 
      nn.BatchNorm3d(midc),
      nn.ReLU(inplace=True),
      nn.Conv3d(midc, outc, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
      nn.BatchNorm3d(outc),
      nn.ReLU(inplace=True)
    )
  
  def forward(self, x):
    return self.double_conv(x)



class Down(nn.Module):
  # (maxpool for (h, w) not t => DoubleConv)
  def __init__(self, inc, outc):
    super(Down, self).__init__()
    self.maxpool_conv = nn.Sequential(
      nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)),
      DoubleConv(inc, outc)
    )

  def forward(self, x):
    # x.size() = (batch, inc, d, h, w) => (batch, outc, d, h//2, w//2)
    return self.maxpool_conv(x)



class Up(nn.Module):
  # (ConvTranspose3d for (h, w), not d => DoubleConv)
  def __init__(self, inc, outc):
    super(Up, self).__init__()

    self.up = nn.ConvTranspose3d(inc, inc//2, kernel_size=(1,2,2), stride=(1,2,2))
    self.conv = DoubleConv(inc, outc)

  def forward(self, x1, x2):
    # x1.size() = (batch, inc//2, d, h, w)
    # x2.size() = (batch, inc//2, d, h, w)
    x1 = self.up(x1)

    diffD = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    diffY = x2.size()[4] - x1.size()[4]

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
    
    x1p = F.pad(x1, padding_list)

    x = torch.cat([x2, x1p], dim=1)

    return self.conv(x)


class OutConv(nn.Module):
  # last conv which maps to the same dimension of the target
  # use point wise convolution (kernel_size=1, padding=0, stride=1)
  def __init__(self, inc, outc):
    super(OutConv, self).__init__()
    self.conv = nn.Conv3d(inc, outc, kernel_size=1)

  def forward(self, x):
    # x.size() = (batch, inc, d, h, w)
    # self.conv(x).size() = (batch, outc, d, h, w)
    return self.conv(x)


class CHWGlobalAveragePooling(nn.Module):
  def __init__(self):
    super(CHWGlobalAveragePooling, self).__init__()
    
  def forward(self, x):
    # x.size() = (batch, c, d, h, w) 

    # x_t.size() = (batch, d, c, h, w)
    x_t = torch.transpose(x, 1, 2)

    # test1.size() = (batch, d, 1, 1, 1)
    test1 = F.avg_pool3d(x_t, kernel_size=x_t.size()[2:])

    # test2.size() = (batch, d)
    test2 = test1.view(x.size(0), -1)

    return test2


class DAttention(nn.Module):
  def __init__(self):
    super(DAttention, self).__init__()
    self.chwgap = CHWGlobalAveragePooling()
  
  def forward(self, x):
    # daverage.size() = (batch, d)
    daverage = self.chwgap(x)

    assert daverage.size(1) == x.size(2)

    tmp = torch.zeros_like(x)
    
    for i in range(x.size(0)):
      for j in range(x.size(2)):
        tmp[i, :, j, :, :] = x[i, :, j, :, :] * daverage[i][j]

    return tmp
    




class UNet3dDAttn(nn.Module):
  '''
    n_channels = c of input = 3
    n_classes = c of target = 1
  '''

  def __init__(self, n_channels, n_classes):
    super(UNet3dDAttn, self).__init__()
    self.n_channels = n_channels
    self.n_classes = n_classes

    self.d_attention = DAttention()

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
    self.sigmoid = nn.Sigmoid()

    
  def forward(self, x):
    x1 = self.inc(x)
    x2 = self.d_attention(self.down1(x1))
    x3 = self.d_attention(self.down2(x2))
    x4 = self.d_attention(self.down3(x3))
    x5 = self.d_attention(self.down4(x4))

    x = self.d_attention(self.up1(x5, x4))
    x = self.d_attention(self.up2(x, x3))
    x = self.d_attention(self.up3(x, x2))
    x = self.d_attention(self.up4(x, x1))

    logits = self.outc(x)
    return self.sigmoid(logits)
    

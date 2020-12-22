import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, bias, padding=None):
    super(ConvLSTMCell, self).__init__()

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    if padding is None:
      self.padding = kernel_size[0]//2, kernel_size[1]//2
    else:
      self.padding = padding

    self.bias = bias

    self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                          out_channels=4*self.hidden_dim,
                          kernel_size=self.kernel_size,
                          padding=self.padding,
                          bias=self.bias,
                          )

  
  def forward(self, input_tensor, cur_state):
    h_cur, c_cur = cur_state

    # input_tensor.size() = (batch, c_of_input_tensor, h, w)
    # h_cur.size() = (batch, hidden_dim, h, w)
    # combined.size() = (batch, c_of_input_tesor_hidden_dim, h, w)
    combined = torch.cat([input_tensor, h_cur], dim=1)
    
    # combined_conv.size() = (batch, 4*hidden_dim, h, w) 
    combined_conv = self.conv(combined)
    
    # cc_i.size() = cc_o.size() = cc_f.size() = cc_g.size() = (batch, hidden_dim, h, w)
    cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

    i = torch.sigmoid(cc_i)
    f = torch.sigmoid(cc_f)
    o = torch.sigmoid(cc_o)
    g = torch.sigmoid(cc_g)

    c_next = f * c_cur + i*g
    h_next = o * torch.tanh(c_next)
    
    # h_next.size() = (batch, hidden_dim, h, w)
    # c_next.size() = (batch, hidden_dim, h, w) 
    return h_next, c_next


  def init_hidden(self, batch_size, image_size):
    height, width = image_size
    return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
            )

     



class ConvLSTM(nn.Module):
  def __init__(self, input_dim, hidden_dim, kernel_size, padding=None, num_layers=3,
              batch_first=False, bias=True, return_all_layers=False):
    super(ConvLSTM, self).__init__()

    self._check_kernel_size_consistency(kernel_size)

    kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
    hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
    if not len(kernel_size) == len(hidden_dim) == num_layers:
      raise

    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.kernel_size = kernel_size
    self.num_layers = num_layers
    self.batch_first = batch_first
    self.bias = bias
    self.return_all_layers = return_all_layers
    self.padding = padding

    cell_list = []
    for i in range(0, self.num_layers):
      cur_input_dim = self.input_dim if i==0 else self.hidden_dim[i-1]
      cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                    hidden_dim=self.hidden_dim[i],
                                    kernel_size=self.kernel_size[i],
                                    bias=self.bias, padding=self.padding))

    
    self.cell_list = nn.ModuleList(cell_list)



  def forward(self, input_tensor, hidden_state=None):
    if not self.batch_first:
      # (t, b, c, h, w) -> (b, t, c, h, w)
      input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
    
    b, _, _, h, w = input_tensor.size()

    if hidden_state is not None:
      raise
    else:
      hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
    

    layer_output_list = []
    last_state_list = []

    seq_len = input_tensor.size(1)
    cur_layer_input = input_tensor

    for layer_idx in range(self.num_layers):
      h, c = hidden_state[layer_idx]
      output_inner = []
      for t in range(seq_len):
        h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                          cur_state=[h,c])
        output_inner.append(h)

      layer_output = torch.stack(output_inner, dim=1)
      cur_layer_input = layer_output

      layer_output_list.append(layer_output)
      last_state_list.append([h, c])

    if not self.return_all_layers:
      layer_output_list = layer_output_list[-1:]
      last_state_list = last_state_list[-1:]

    return layer_output_list, last_state_list


  def _init_hidden(self, batch_size, image_size):
    init_states = []
    for i in range(self.num_layers):
      init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
    return init_states


  @staticmethod
  def _check_kernel_size_consistency(kernel_size):
    if not (isinstance(kernel_size, tuple) or (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
      raise

  @staticmethod
  def _extend_for_multilayer(param, num_layers):
    if not isinstance(param, list):
      param = [param] * num_layers
    return param


class DoubleConv(nn.Module):
  def __init__(self, inc, outc, midc=None):
    super(DoubleConv, self).__init__()
    self.inc = inc
    self.outc = outc
    if not midc:
      midc = outc
    self.double_conv = nn.Sequential(
      nn.Conv2d(inc, midc, kernel_size=(3,3), padding=(1,1)),
      nn.BatchNorm2d(midc),
      nn.ReLU(inplace=True),
      nn.Conv2d(midc, outc, kernel_size=(3,3), padding=(1,1)),
      nn.BatchNorm2d(outc),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    # x.size() = (batch, t, inc, h, w)
    # x2.size() = (batch*t, inc, h, w)
    x2 = x.view(-1, x.size(2), x.size(3), x.size(4))
    print("here1")
    print(x2.size())
    print(self.inc)
    print(self.outc)
    # x3.size() = (batch*t, outc, h, w)
    x3 = self.double_conv(x2)

    # x4.size() = (batch, t, outc, h, w)
    x4 = x3.view(x.size(0), x.size(1), self.outc ,x.size(3), x.size(4))
    return x4


#ConvLSTM(input_dim, hidden_dim, kernel_size, padding, num_layers, batch_first, bias, return_all_layers)

class Down(nn.Module):
  def __init__(self, inc, outc):
    super(Down, self).__init__()
    self.inc = inc
    self.outc = outc
    self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
    self.lstmconv = ConvLSTM(input_dim=inc, hidden_dim=outc, kernel_size=(3,3), padding=(1,1),
                              num_layers=1, batch_first=True, bias=False, return_all_layers=True)

  def forward(self, x):
    # x.size() = (batch, t, inc, h, w)
    # x2.size() = (batch*t, inc, h, w)
    x2 = x.view(-1, x.size(2), x.size(3), x.size(4))
    # x3.size() = (batch*t, inc, h//2, w//2)
    x3 = self.maxpool(x2)
    # x4.size() = (batch, t, inc, h//2, w//2)
    
    x4 = x3.view(x.size(0), x.size(1), x.size(2), x3.size(2), x3.size(3))
    print("****"*100)
    print(x4.size())
    print(self.inc)
    print(self.outc)
    lstm_output1, lstm_output2 = self.lstmconv(x4)
    
    # lstm_output1 has length == num_layers
    # lstm_output1[0].size() = (batch, t, outc, h//2, w//2)

    return lstm_output1[0]
      

class Up(nn.Module):
  def __init__(self, inc, outc):
    super(Up, self).__init__()
    self.up = nn.ConvTranspose2d(inc, inc//2, kernel_size=(2,2), stride=(2,2))
    self.lstmconv = ConvLSTM(input_dim=inc, hidden_dim=outc, kernel_size=(3,3), padding=(1,1),
                             num_layers=1, batch_first=True, bias=False, return_all_layers=True)

  def forward(self, x1, x2):
    # x1.size() = (batch, t, inc, h//2, w//2)
    # x2.size() = (batch, t, inc//2, h, w)
    
    # x1a.size() = (batch * t, inc, h//2, w//2)
    # x2a.size() = (batch * t, inc//2, h, w)
    x1a = x1.view(-1, x1.size(2), x1.size(3), x1.size(4))
    x2a = x2.view(-1, x2.size(2), x2.size(3), x2.size(4))
    # x1b.size() = (batch * t, inc//2, h, w)
    x1b = self.up(x1a)

    diffY = x2a.size(2) - x1b.size(2)
    diffX = x2a.size(3) - x1b.size(3)

    x1b = F.pad(x1b, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
    
    # x.size() = (batch * t, inc, h, w)
    x = torch.cat([x2a, x1b], dim=1)
    
    # x.size() = (batch, t, inc, h, w)
    x = x.view(x1.size(0), x1.size(1), x.size(1), x.size(2), x.size(3))

    lstm_output1, lstm_output2 = self.lstmconv(x)

    # lstm_output1 has length == num_layers
    # lstm_output1[0].size() = (batch, t, outc, h, w)
    return lstm_output1[0]



class OutConv(nn.Module):
  # point wise conv to map to the target space
  def __init__(self, inc, outc):
    self.inc = inc
    self.outc = outc
    super(OutConv, self).__init__()
    self.conv = nn.Conv2d(inc, outc, kernel_size=1) 

  def forward(self, x):
    # x.size() = (batch, t, inc, h, w)
    # x2.size() = (batch * t, inc, h, w)
    x2 = x.view(-1, x.size(2), x.size(3), x.size(4))
    # x3.size() = (batch * t, outc, h, w)
    print("before outconv")
    print(x2.size())
    x3 = self.conv(x2)
    # x4.size() = (batch, t, outc, h, w)
    x4 = x3.view(x.size(0), x.size(1), self.outc, x.size(3), x.size(4))
    return x4 





class LSTMUNet(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(LSTMUNet, self).__init__()
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

    self.sigmoid = nn.Sigmoid()


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

    return self.sigmoid(logits)
    



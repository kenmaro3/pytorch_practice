import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary


class MyModel(nn.Module):
  def __init__(self, n_inputs,  n_classes):
    super(MyModel, self).__init__()
    model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
    for param in model.parameters():
      param.requires_gred = False
    self.original_model = model

    self.original_model.classifier = nn.Sequential(
      nn.Linear(n_inputs, 2048),
      nn.ReLU(),
      nn.Dropout(0.4),
      nn.Linear(2048, n_classes),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.original_model(x)


if __name__ == "__main__":
  model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

  #summary(model, (3,224,224))

  #print(dir(model))
  #model.eval()

  for param in model.parameters():
    param.requires_grad = False

  model.classifier = nn.Sequential(
    nn.Linear(9216, 2048),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(2048, 4),
    #nn.LogSoftmax(dim=1)
  )
  print(model)


  #model_ft = model

  #model = MyModel(9216, 2)
  #criterion = nn.NLLLoss()
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters())

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  model = model.to(device)

  x = torch.randn(1, 3, 128, 128)
  target = torch.LongTensor([1,0])
  output = model(x)

  print(output.size())





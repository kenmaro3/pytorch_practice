import torch

def train(model, train_loader,optimizer, criterion, device, log_interval, epoch_idx):
    model.train()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        if i % log_interval == 0:
            print(
                    "train epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f}".format(
                        epoch_idx,
                        i * len(data),
                        len(train_loader.dataset),
                        100.0 * i /len(train_loader),
                        loss.item()
                        )
            )

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0

    conf_mat = dict(
      l0t = 0,
      l1t = 0,
      l2t = 0,
      l3t = 0,
      l0c = 0,
      l1c = 0,
      l2c = 0,
      l3c = 0,

    )

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)

            prediction = output.argmax(dim=1, keepdim=True)

            test = target.view_as(prediction)

            for i in range(test.size(0)):
              if test[i] == 0:
                conf_mat['l0t'] += 1
                conf_mat['l0c'] += prediction[i].eq(test[i]).item()
              elif test[i] == 1:
                conf_mat['l1t'] += 1
                conf_mat['l1c'] += prediction[i].eq(test[i]).item()
              elif test[i] == 2:
                conf_mat['l2t'] += 1
                conf_mat['l2c'] += prediction[i].eq(test[i]).item()
              elif test[i] == 3:
                conf_mat['l3t'] += 1
                conf_mat['l3c'] += prediction[i].eq(test[i]).item()

            correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)

        print(
            "\nTest set: average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset)
                )
        )

    print(conf_mat)

    print(f'l0 >>> {conf_mat["l0c"]/conf_mat["l0t"]}')
    print(f'l1 >>> {conf_mat["l1c"]/conf_mat["l1t"]}')
    print(f'l2 >>> {conf_mat["l2c"]/conf_mat["l2t"]}')
    print(f'l3 >>> {conf_mat["l3c"]/conf_mat["l3t"]}')


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

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
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
                100.0 * correct / len(test_loader.dataset)
                )
        )


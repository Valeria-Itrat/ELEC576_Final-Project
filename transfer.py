import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class EmbeddingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

def train(epoch,model,criterion,optimizer,train_loader,device,logging_interval=10,writer=None):
    model.train()
    correct = 0
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        train_loss += loss.item() * data.size(0) 

        if batch_idx % logging_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
          if writer:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Batch_Loss", loss.item(), global_step)

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    if writer:
        writer.add_scalar(f'Train Loss', train_loss, epoch)
        writer.add_scalar(f'Train Accuracy', train_accuracy, epoch)
    return train_loss, train_accuracy

def test(epoch,model,device,test_loader,criterion,writer=None):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += criterion(output, target).item() * data.size(0)

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),test_accuracy))
    if writer:
        writer.add_scalar(f'Test Loss', test_loss, epoch)
        writer.add_scalar(f'Test Accuracy', test_accuracy, epoch)
    return test_loss, test_accuracy
import torch.nn as nn
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


######################################################
def extract_pann_embedding(audio_path, model, device, win_sec=1.0, hop_sec=0.5, sr=16000):
    """
    extract embedding but keeping temporal patterns

    """
    waveform, sr = librosa.load(audio_path, sr=16000)
    n_samples = len(waveform)

    win_size = int(win_sec * sr)
    hop_size = int(hop_sec * sr)

    emb_list = []
    for start in range(0, n_samples - win_size + 1, hop_size):
        chunk = waveform[start:start + win_size]
        chunk_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(chunk_tensor,None)
            emb = output["embedding"].squeeze(0).cpu().numpy()
        emb_list.append(emb)
    if len(emb_list) == 0:
        return None 
    return np.stack(emb_list)

def check_emb_lengths(final_df):
  """
  Make all the embeddings same size

  """
  def check_emb(final_df):
    emb_lengths = []
    for path in final_df['embedding_path']:
        emb = np.load(path)
        emb_lengths.append(emb.shape[0])
    max_m = max(emb_lengths)
    min_m = min(emb_lengths)
    print("Max", max_m, 'Min:',min_m)
    return max_m
  max_m = check_emb(final_df)
  # correct mel lenght, I use pading to make all the same size
  for path in final_df['embedding_path']:
    emb = np.load(path)
    T, D = emb.shape

    if T < max_m:
      pad_cant = max_m - T
      pad = np.zeros((pad_cant, D), dtype=np.float32)
      emb = np.concatenate([emb, pad], axis=0)
    np.save(path, emb)
  check_emb(final_df)

######################################################
class PANN_LSTM(nn.Module):
    def __init__(self, num_classes=2, input_dim=2048, hidden_size=256, num_layers=2, dropout_rate=0.5, bi=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional= bi
        )
        self.dropout = nn.Dropout(dropout_rate)
        if bi:
            self.fc = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out,_ = self.lstm(x)
        last = self.dropout(out[:,-1,:])
        return self.fc(last)

class Net(nn.Module):
    def __init__(self, num_classes=2, input_dim=2048, hidden_size=256, num_layers=2, dropout_rate=0.5, RNN_TYPE='LSTM'):
        super(Net, self).__init__()

        # https://github.com/pytorch/examples/blob/main/word_language_model/model.py
        if RNN_TYPE in ['LSTM', 'GRU']:
          self.rnn = getattr(nn, RNN_TYPE)(input_dim, hidden_size, num_layers,
                                           batch_first=True, dropout= dropout_rate)
        else:
          self.rnn = nn.RNN(input_size=input_dim,
                            hidden_size= hidden_size, num_layers= num_layers,
                            batch_first=True, dropout= dropout_rate)

        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        r_out, _ = self.rnn(x, None) 
        out = self.out(r_out[:, -1, :])
        return out


######################################################
def aug_noise(emb, noise_level=0.01):
    # Gaussian noise
    return emb + np.random.normal(0, noise_level, emb.shape).astype(np.float32)
def aug_time_mask(emb, p=0.05):
    # silence masks 
    emb = emb.copy()
    T = emb.shape[0]
    mask_len = int(T * p)
    start = np.random.randint(0, max(1, T - mask_len))
    emb[start:start+mask_len] = 0
    return emb
def aug_dropout(emb, drop_prob=0.03):
    # random dropout
    mask = np.random.rand(*emb.shape) > drop_prob
    return emb * mask
def augment_embedding(path, n_aug=2):
    """
    Data augmentation using different functins randomly
    """
    emb = np.load(path)
    augmented = []
    for _ in range(n_aug):
        e = emb.copy()
        if np.random.rand() < 0.7:
            e = aug_noise(e)
        if np.random.rand() < 0.5:
            e = aug_time_mask(e)
        if np.random.rand() < 0.5:
            e = aug_dropout(e)
        augmented.append(e)
    return augmented

######################################################
class EmbeddingDataset2(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

class EmbeddingDatase(Dataset):
    def __init__(self, df):
        self.paths = df["embedding_path"].values
        self.labels = df["label_idx"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        emb = np.load(self.paths[idx]).astype(np.float32)
        label = int(self.labels[idx])
        return torch.tensor(emb), torch.tensor(label)
    
###########################################
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
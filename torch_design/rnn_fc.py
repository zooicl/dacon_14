import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RNN_FC_Model(torch.nn.Module):
    def __init__(self, input_size, vocab_size, embed_size, hidden_size, num_layers, device):
        super(RNN_FC_Model, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size,
                                  hidden_size,
                                  num_layers,
                                  batch_first=True,
                                  bidirectional=False,
                                  #                      dropout=0.3
                                  )

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512), self.relu, torch.nn.BatchNorm1d(512), self.dropout,
            torch.nn.Linear(512, 256), self.relu, torch.nn.BatchNorm1d(256), self.dropout,
            torch.nn.Linear(256, 128), self.relu, torch.nn.BatchNorm1d(128), self.dropout,
        )

        self.output = torch.nn.Linear(hidden_size * 2 + 128, 2)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, input_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)

        return hidden, cell

    def forward(self, x, seq):
        embed = self.embed(seq)

        hidden, cell = self.init_hidden(embed.size(0))  # initial hidden,cell

        out, (hidden, cell) = self.lstm(embed, (hidden, cell))

        hidden = hidden[-1:]
        merged = torch.cat([h for h in hidden] + [self.fc(x)], 1)

        return self.output(merged)


class RNN_FC_Dataset(Dataset):
    def __init__(self, df, y_col, seq_col='idx'):
        self.seq_col = seq_col
        self.cols = [c for c in df.columns if c not in [y_col, seq_col]]

        print(seq_col, len(self.cols), y_col)

        self.X = df[self.cols].values
        self.y = pd.get_dummies(df[y_col].astype(int), prefix=y_col).values

        self.seq_X = np.stack(df[self.seq_col].values)

    def __len__(self):
        return len(self.X)

    def get_feature_names(self):
        return self.cols + [self.seq_col]

    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        X_seq = self.seq_X[idx].astype(np.int64)
        y = self.y[idx].astype(np.float32)

        #         print(X.shape, X_seq.shape, y.shape)
        return X, X_seq, y


def train_torch(model, dataset, criterion, optimizer, scheduler, device, step=100, num_workers=3):
    model.train()
    loss = 0
    acc = 0
    data_loader = DataLoader(dataset=dataset,
                             #                           batch_size=100000,
                             #                         batch_size=int(train_size * 0.7),
                             batch_size=len(dataset) // step,
                             #                           batch_size=10000,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True
                             )
    for i, data in enumerate(data_loader):
        #     for i, data in tqdm_notebook(enumerate(train_loader), total=len(train_loader), desc = 'epoch{}_batch'.format(e)):
        X_batch, X_seq_batch, y_batch = data

        X_batch = X_batch.to(device)
        X_seq_batch = X_seq_batch.to(device)
        y_batch = y_batch.to(device)

        #         print(X_batch.size())

        y_pred = model(X_batch, X_seq_batch)

        loss = criterion(y_pred, y_batch)

        loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()

    return loss / len(dataset), acc / len(dataset)


def test_torch(model, dataset, criterion, device, step=100, num_workers=3):
    model.eval()
    loss = 0
    acc = 0

    y_true_list = []
    y_score_list = []

    data_loader = DataLoader(dataset=dataset,
                             batch_size=len(dataset) // step,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=True
                             )

    for i, data in enumerate(data_loader):
        X_batch, X_seq_batch, X_rev_seq_batch, y_batch = data
        y_true = y_batch

        X_batch = X_batch.to(device)
        X_seq_batch = X_seq_batch.to(device)
        y_batch = y_batch.to(device)
        y_true_list.append(y_true[:, 1].cpu().detach().numpy())

        with torch.no_grad():
            y_pred = model(X_batch, X_seq_batch)
            loss = criterion(y_pred, y_batch)
            loss += loss.item()
            acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()

            y_pred = torch.sigmoid(y_pred)
            y_score_list.append(y_pred[:, 1].cpu().detach().numpy())

    return loss / len(dataset), acc / len(dataset), np.concatenate(y_true_list, axis=0), np.concatenate(y_score_list,
                                                                                                        axis=0)

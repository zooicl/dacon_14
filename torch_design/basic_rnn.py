import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RNNModel(torch.nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers, device):
        super(RNNModel, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embed = torch.nn.Embedding(input_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size,
                                  hidden_size,
                                  num_layers,
                                  batch_first=True,
                                  bidirectional=False)
        self.linear = torch.nn.Linear(hidden_size, 64)
        self.linear64 = torch.nn.Linear(64, 2)

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.7)

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, input_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(self.device)
        return hidden, cell

    def forward(self, x):
        embed = self.embed(x)

        hidden, cell = self.init_hidden(embed.size(0))  # initial hidden,cell
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))

        hidden = hidden[-1:]
        hidden = torch.cat([h for h in hidden], 1)
        output = self.linear(hidden)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear64(output)
        return output


# class RNNDataset(Dataset):
#     def __init__(self, df, y_col, fc_cols, seq_cols):
#         self.fc_cols = fc_cols
#         self.seq_cols = seq_cols
#
#         self.fc_X = None
#         if fc_cols:
#             self.fc_X = df[fc_cols].values
#
#         self.seq_X = []
#         for c in seq_cols:
#             self.seq_X.append(np.stack(df[c].values))
#
#         self.y = pd.get_dummies(df[y_col].astype(int), prefix=y_col).values
#
#     def __len__(self):
#         return len(self.y)
#
#     def get_feature_names(self):
#         return [self.fc_cols] + [self.seq_cols]
#
#     def __getitem__(self, idx):
#         X_batch_list = []
#         if self.fc_X:
#             X_batch_list.append(self.fc_X[idx].astype(np.float32))
#
#         for seq in self.seq_X:
#             X_batch_list.append(seq[idx].astype(np.int64))
#
#         return X_batch_list, self.y[idx].astype(np.float32)
#
#
# def train_torch(model, dataset, criterion, optimizer, scheduler, device, step=100, num_workers=3):
#     model.train()
#     loss = 0
#     acc = 0
#     data_loader = DataLoader(dataset=dataset,
#                              #                           batch_size=100000,
#                              #                         batch_size=int(train_size * 0.7),
#                              batch_size=len(dataset) // step,
#                              #                           batch_size=10000,
#                              shuffle=True,
#                              num_workers=num_workers,
#                              drop_last=True
#                              )
#     for i, data in enumerate(data_loader):
#         X_seq_batch, y_batch = data
#
#         X_seq_batch = X_seq_batch.to(device)
#         y_batch = y_batch.to(device)
#
#         y_pred = model(X_seq_batch)
#
#         loss = criterion(y_pred, y_batch)
#
#         loss += loss.item()
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()
#
#     return loss / len(dataset), acc / len(dataset)
#
#
# def test_torch(model, dataset, criterion, device, step=100, num_workers=3):
#     model.eval()
#     loss = 0
#     acc = 0
#
#     y_true_list = []
#     y_score_list = []
#
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=len(dataset) // step,
#                              shuffle=False,
#                              num_workers=num_workers,
#                              drop_last=True
#                              )
#
#     for i, data in enumerate(data_loader):
#         X_batch, X_seq_batch, X_rev_seq_batch, y_batch = data
#         y_true = y_batch
#
#         X_batch = X_batch.to(device)
#         X_seq_batch = X_seq_batch.to(device)
#         X_rev_seq_batch = X_rev_seq_batch.to(device)
#         y_batch = y_batch.to(device)
#         y_true_list.append(y_true[:, 1].cpu().detach().numpy())
#
#         with torch.no_grad():
#             y_pred = model(X_batch, X_seq_batch, X_rev_seq_batch)
#             loss = criterion(y_pred, y_batch)
#             loss += loss.item()
#             acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()
#
#             y_pred = torch.sigmoid(y_pred)
#             y_score_list.append(y_pred[:, 1].cpu().detach().numpy())
#
#     return loss / len(dataset), acc / len(dataset), np.concatenate(y_true_list, axis=0), np.concatenate(y_score_list,
#                                                                                                         axis=0)

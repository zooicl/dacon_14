import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class NN_Model(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, device):
        super(NN_Model, self).__init__()
        self.device = device

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_layers[0]), self.relu, torch.nn.BatchNorm1d(hidden_layers[0]),
            self.dropout,
        )

        for i in range(0, len(hidden_layers) - 1):
            self.fc.add_module(f'hidden_{i}_{i + 1}', torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.fc.add_module(f'bn_{i}_{i + 1}', torch.nn.BatchNorm1d(hidden_layers[i + 1]))
            self.fc.add_module(f'dropout_{i}_{i + 1}', self.dropout)
        self.fc.add_module(f'hidden_last', torch.nn.Linear(hidden_layers[-1], 2))

    def forward(self, x):
        return self.fc(x[0])


class NN_Dataset(Dataset):
    def __init__(self, df, y_col):
        self.cols = [c for c in df.columns if c not in [y_col]]
        self.X = df[self.cols].values
        self.y = pd.get_dummies(df[y_col].astype(int), prefix=y_col).values

    def __len__(self):
        return len(self.X)

    def get_feature_names(self):
        return self.cols

    def __getitem__(self, idx):
        X = self.X[idx].astype(np.float32)
        y = self.y[idx].astype(np.float32)

        #         print(X.shape, X_seq.shape, y.shape)
        return [X], y
#
#
# def train_torch(model, dataset, criterion, optimizer, scheduler, device, step=100, num_workers=3):
#     model.train()
#     loss = 0
#     acc = 0
#     data_loader = DataLoader(dataset=dataset,
#                              batch_size=len(dataset) // step,
#                              shuffle=True,
#                              num_workers=num_workers,
#                              drop_last=True
#                              )
#     for i, data in enumerate(data_loader):
#         #     for i, data in tqdm_notebook(enumerate(train_loader), total=len(train_loader), desc = 'epoch{}_batch'.format(e)):
#         X_batch_list, y_batch = data
#
#         for i in range(len(X_batch_list)):
#             X_batch_list[i] = X_batch_list[i].to(device)
#
#         y_batch = y_batch.to(device)
#
#         y_pred = model(X_batch_list)
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
#         X_batch_list, y_batch = data
#         y_true = y_batch
#
#         for i in range(len(X_batch_list)):
#             X_batch_list[i] = X_batch_list[i].to(device)
#
#         y_batch = y_batch.to(device)
#         y_true_list.append(y_true[:, 1].cpu().detach().numpy())
#
#         with torch.no_grad():
#             y_pred = model(X_batch_list)
#             loss = criterion(y_pred, y_batch)
#             loss += loss.item()
#             acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()
#
#             y_pred = torch.sigmoid(y_pred)
#             y_score_list.append(y_pred[:, 1].cpu().detach().numpy())
#
#     return loss / len(dataset), acc / len(dataset), np.concatenate(y_true_list, axis=0), np.concatenate(y_score_list,
#                                                                                                         axis=0)

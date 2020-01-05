import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KBDataset(Dataset):
    def __init__(self, df, y_col, fc_cols, seq_cols):
        self.fc_cols = fc_cols
        self.seq_cols = seq_cols

        self.fc_X = None
        if fc_cols:
            self.fc_X = df[fc_cols].values

        self.seq_X = []
        for c in seq_cols:
            self.seq_X.append(np.stack(df[c].values))

        self.y = pd.get_dummies(df[y_col].astype(int), prefix=y_col).values

    def __len__(self):
        return len(self.y)

    def get_feature_names(self):
        return [self.fc_cols] + [self.seq_cols]

    def __getitem__(self, idx):
        X_batch_list = []
        if self.fc_X:
            X_batch_list.append(self.fc_X[idx].astype(np.float32))

        for seq in self.seq_X:
            X_batch_list.append(seq[idx].astype(np.int64))

        return X_batch_list, self.y[idx].astype(np.float32)


def train_torch(model, dataset, criterion, optimizer, scheduler, device, step=100, num_workers=3):
    model.train()
    loss = 0
    acc = 0
    data_loader = DataLoader(dataset=dataset,
                             batch_size=len(dataset) // step,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True
                             )
    for i, data in enumerate(data_loader):
        #     for i, data in tqdm_notebook(enumerate(train_loader), total=len(train_loader), desc = 'epoch{}_batch'.format(e)):
        X_batch_list, y_batch = data

        for i in range(len(X_batch_list)):
            X_batch_list[i] = X_batch_list[i].to(device)

        y_batch = y_batch.to(device)

        y_pred = model(*X_batch_list)

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
        X_batch_list, y_batch = data
        y_true = y_batch

        for i in range(len(X_batch_list)):
            X_batch_list[i] = X_batch_list[i].to(device)

        y_batch = y_batch.to(device)
        y_true_list.append(y_true[:, 1].cpu().detach().numpy())

        with torch.no_grad():
            y_pred = model(*X_batch_list)
            loss = criterion(y_pred, y_batch)
            loss += loss.item()
            acc += (y_pred.argmax(1) == y_batch.argmax(1)).sum().item()

            y_pred = torch.sigmoid(y_pred)
            y_score_list.append(y_pred[:, 1].cpu().detach().numpy())

    return loss / len(dataset), acc / len(dataset), np.concatenate(y_true_list, axis=0), np.concatenate(y_score_list,
                                                                                                        axis=0)

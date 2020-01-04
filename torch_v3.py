#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import time
import numpy as np
from datetime import datetime
from sklearn.externals import joblib 
import os
from konlpy.tag import Mecab
import lightgbm as lgb
print(lgb.__version__)

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
from sklearn.model_selection import StratifiedKFold

import gc

from tqdm import tqdm_notebook

import warnings
warnings.filterwarnings(action='ignore')


import torch
print(torch.__version__)
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from torchsummary import summary

from tools import EarlyStopping, eval_summary

print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
device


# #### Model

# In[2]:


class RDModel(torch.nn.Module):
    def __init__(self, input_size, vocab_size, embed_size, hidden_size, num_layers):
        super(RDModel,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.25)
        
#         self.embed = torch.nn.Embedding(vocab_size, embed_size, sparse=True)
        self.embed = torch.nn.Embedding(vocab_size, embed_size)
        self.lstm = torch.nn.LSTM(embed_size,
                    hidden_size,
                    num_layers,
                    batch_first=True,
                    bidirectional=False, 
                     dropout=0.3)
        
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(input_size, 2048), self.relu, torch.nn.BatchNorm1d(2048), self.dropout,
#             torch.nn.Linear(2048, 1024), self.relu, torch.nn.BatchNorm1d(1024), self.dropout,
# #             torch.nn.Linear(1024, 512), self.relu, torch.nn.BatchNorm1d(512), self.dropout,
# #             torch.nn.Linear(512, 512), self.relu, torch.nn.BatchNorm1d(512), self.dropout,
# #             torch.nn.Linear(512, 256), self.relu, torch.nn.BatchNorm1d(256), self.dropout,
# #             torch.nn.Linear(256, 128), self.relu, torch.nn.BatchNorm1d(128), self.dropout,
            
#             torch.nn.Linear(1024, 128), self.relu, torch.nn.BatchNorm1d(128), self.dropout,
# #             torch.nn.Linear(128, 2), 
#         )
#         self.output = torch.nn.Linear(hidden_size + 128, 2)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512), self.relu, torch.nn.BatchNorm1d(512), self.dropout,
            torch.nn.Linear(512, 256), self.relu, torch.nn.BatchNorm1d(256), self.dropout,
            torch.nn.Linear(256, 128), self.relu, torch.nn.BatchNorm1d(128), self.dropout,
        )
        
        self.output = torch.nn.Linear(hidden_size + 128, 2)
        
#         self.output = torch.nn.Linear(hidden_size, 2)
        

    def init_hidden(self, batch_size):
        # (num_layers * num_directions, input_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device)
        cell = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).to(device)
        return hidden, cell

        
    def forward(self, x, seq):   
        embed = self.embed(seq)
        hidden, cell = self.init_hidden(embed.size(0)) # initial hidden,cell
        output, (hidden, cell) = self.lstm(embed, (hidden, cell))
        
        hidden = hidden[-1:]
        hidden = torch.cat([h for h in hidden] + [self.fc(x)], 1)
#         hidden = torch.cat([h for h in hidden], 1)
        
        return self.output(hidden)


# #### Dataset

# In[3]:


class RDDataset(Dataset):
    def __init__(self, df, y_col, seq_col='text_idx'):
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


# #### train/test_torch

# In[4]:


from torch.utils.data import DataLoader

def train_torch(dataset, step=100, num_workers=3):
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
        
        del X_batch, y_batch, y_pred
        gc.collect()

    return loss / len(dataset), acc / len(dataset)


def test_torch(dataset, step=100, num_workers=3):
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
        X_batch, X_seq_batch, y_batch = data
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
            
#              del X_batch, y_batch, y_true, y_pred
            
    return loss / len(dataset), acc / len(dataset), np.concatenate(y_true_list, axis=0), np.concatenate(y_score_list, axis=0)


# In[5]:


fc_cols = [
#     'tfidf_pos_word_22_0028',
#  'tfidf_pos_char_11_0000',
#  'tfidf_word_11_1263',
#  'tfidf_word_11_1516',
#  'tfidf_word_11_0552',
#  'cnt_0583',
#  'tfidf_word_22_0130',
#  'tfidf_word_11_0177',
#  'tfidf_word_11_0307',
#  'tfidf_word_22_0132',
#  'tfidf_word_11_0928',
#  'tfidf_word_11_0186',
#  'cnt_0492',
#  'tfidf_pos_word_11_0129',
#  'tfidf_char_11_0264',
#  'tfidf_pos_char_11_0650',
#  'tfidf_pos_char_11_0242',
#  'tfidf_char_11_0731',
#  'tfidf_word_11_0916',
#  'tfidf_pos_char_11_0213',
#  'tfidf_pos_word_22_0021',
#  'tfidf_char_11_0230',
#  'tfidf_pos_word_11_0391',
#  'cnt_0041',
#  'cnt_0042',
#  'tfidf_char_11_0796',
#  'tfidf_word_22_0095',
#  'tfidf_word_11_0011',
#  'tfidf_word_11_0736',
#  'tfidf_pos_char_11_0005',
#  'tfidf_pos_word_11_0077',
#  'fea__noun',
#  'cnt_0126',
#  'cnt_0223',
#  'tfidf_word_11_1439',
#  'tfidf_pos_char_11_0003',
#  'tfidf_word_11_0854',
#  'tfidf_word_11_1660',
#  'tfidf_char_11_0359',
#  'tfidf_pos_char_11_0589',
#  'cnt_0715',
#  'tfidf_pos_char_11_0415',
#  'tfidf_pos_word_11_0235',
#  'tfidf_char_11_0702',
#  'tfidf_char_11_0464',
#  'tfidf_pos_char_11_0017',
#  'tfidf_word_11_0319',
#  'tfidf_pos_char_11_0626',
#  'tfidf_pos_word_11_0420',
#  'tfidf_char_11_0657',
#  'tfidf_word_22_0091',
#  'cnt_0796',
#  'tfidf_char_11_0126',
#  'tfidf_word_11_0166',
#  'tfidf_word_33_0026',
#  'fea__text_len',
#  'tfidf_char_11_0130',
#  'tfidf_word_22_0134',
#  'cnt_0007',
#  'tfidf_pos_word_11_0044',
#  'tfidf_pos_char_11_0007',
#  'tfidf_pos_char_11_0324'
          ]


# #### Load Data

# In[6]:


# merged_ts = '20191230T014439_8180'
# merged_ts = '20191229T155539'
# merged_ts = '20191231T113708_5499'
# merged_ts = '20191231T165424_6099'
# merged_ts = '20191231T162533_2022'
# merged_ts = '20200101T212111_5854_100_24161'
merged_ts = '20200102T015155_8438_128_49980'

merged_ts = '20200103T111811_8438_256_1774'

merged_ts = '20200103T112827_8438_256_49980'


# merged_ts = '20200102T113923_8438_744_49980'

# merged_ts = '20200102T160226_8438_275_997'


train_path = 'data/df_merged_{}_train.pkl'.format(merged_ts)
test_path = 'data/df_merged_{}_test.pkl'.format(merged_ts)

df_model = joblib.load(train_path)  
df_model = df_model.reset_index()
print('model_set\n', df_model['smishing'].value_counts())
df_test = joblib.load(test_path) 


# In[7]:


idx_cols = ['smishing', 'id', 'index']

seq_col = [c for c in df_model.columns if '_idx' in c][0]

fea_cols = [c for c in df_model.columns if c not in idx_cols + [seq_col]]

# fea_cols = [c for c in fea_cols if c in fc_cols] + [c for c in df_model.columns if 'fea__' in c]

fea_cols = list(set(fea_cols))

# fea_cols.remove(seq_col)
input_size = len(fea_cols)

vocab_size = int(merged_ts.split('_')[-1])

x_test = torch.Tensor(df_test[fea_cols].values).to(device)
x_seq_test = torch.Tensor(np.stack(df_test[seq_col].values)).long().to(device)


# In[8]:


input_size, len(fea_cols), seq_col, vocab_size


# In[9]:


df_model[fea_cols].shape


# #### Training

# In[10]:


params_dataloader = {
    'step' : 170,
    'num_workers': 2,
}

params_model = {
    'input_size':input_size, 
    'vocab_size':vocab_size,
    'embed_size':128, 
    'hidden_size':128,
    'num_layers':5,
}

print('merged_ts', merged_ts)
print(params_dataloader)
print(params_model)

[df_test.drop(c, axis=1, inplace=True) for c in df_test.columns if 'smishing_' in c]

skf = StratifiedKFold(n_splits=5, random_state=8405)

for cv, index in enumerate(skf.split(df_model[fea_cols], df_model['smishing'])):
    train_index, valid_index = index
    train_set = RDDataset(df_model.loc[train_index, fea_cols + [seq_col, 'smishing']],
                          'smishing', seq_col)
    valid_set = RDDataset(df_model.loc[valid_index, fea_cols + [seq_col, 'smishing']],
                          'smishing', seq_col)
    
    print(len(train_index), len(valid_index))
    print('\nCV', cv)
    model = RDModel(**params_model).to(device)
    
    epoch = 1
    if cv == 0:
#         print(summary(model, (input_size, )))
        print(model.train())
    
    early_stopping = EarlyStopping(patience=5, min_epoch=18, verbose=True)
    
    pos_weight = torch.Tensor([1., 10.,])
#     pos_weight = torch.Tensor([1., 1.,])
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#     optimizer = torch.optim.SparseAdam(model.parameters(), lr = 0.0025)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.999)

    model_ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    print('model_ts', model_ts)
    print('Epoch:', epoch)

    N_EPOCHS = 100
    for e in tqdm_notebook(range(epoch, epoch + N_EPOCHS), total=N_EPOCHS, desc = 'CV {} Epoch'.format(cv)):
        train_loss, train_acc = train_torch(train_set, **params_dataloader)
        valid_loss, valid_acc, y_true, y_score = test_torch(valid_set, **params_dataloader)
        print('[{}] CV {} Epoch {}\n\tTrain loss: {}\tValid loss: {}\t{}'.format(
            datetime.now().strftime('%Y%m%dT%H%M%S'), 
            cv, e, train_loss, valid_loss, train_loss / valid_loss))
        
        eval_dict = eval_summary(y_true, y_score, cut_off=0.5)
        print('\t', eval_dict)
        
        early_stopping(-eval_dict['auc'], model)

        if early_stopping.early_stop:
            print("\tEarly stopping epoch {}, valid loss {}".format(e, valid_loss))
            break
    
        epoch = e + 1
    
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model.state_dict(), 'model/{}_{}_{}.model'.format(model_ts, cv, early_stopping.best_epoch))
    print('\nBest_Epoch', early_stopping.best_epoch)
    
    valid_loss, valid_acc, y_true, y_score = test_torch(valid_set)
    valid_dict = eval_summary(y_true, y_score, cut_off=0.5)
    print('END<valid> CV {} eval summary\n'.format(cv), valid_dict)

    train_loss, train_acc, y_true, y_score = test_torch(train_set)
    train_dict = eval_summary(y_true, y_score, cut_off=0.5)
    print('END<train> CV {} eval summary\n'.format(cv), train_dict)
    
    print('train_auc - valid_auc:', train_dict['auc'] - valid_dict['auc'])

    
    model.eval()
    pred_col = 'smishing_{}'.format(cv)
    df_test[pred_col] = torch.sigmoid(model(x_test, x_seq_test))[:, 1].cpu().detach().numpy()
    df_test[[pred_col]].to_csv('submit/{}_{}_nn.csv'.format(model_ts, pred_col), index=True)
    
    del train_set, valid_set
    
#     break


# #### Predict Train

# In[ ]:


df = pd.Series(y_score)
df.hist(bins=100, figsize=(20, 5))
(df * 10).astype(int).value_counts(sort=False)


# In[ ]:


# df_model[(y_score <= 0.5) & (y_true == 1)]['text']


# In[ ]:


# df_model[(y_score > 0.5) & (y_true == 0)]['text']


# #### Predict Test

# In[ ]:


pred_cols = [c for c in df_test.columns if 'smishing_' in c]
print(len(pred_cols))
df_test['pred_max'] = df_test[pred_cols].max(axis=1)
df_test['pred_min'] = df_test[pred_cols].min(axis=1)
df_test['pred_mean'] = df_test[pred_cols].mean(axis=1)
df_test['pred_std'] = df_test[pred_cols].std(axis=1)

print(df_test['pred_std'].max(), df_test['pred_std'].min(), df_test['pred_std'].mean())

df_test['smishing'] = df_test['pred_mean']


# In[ ]:


df_test['smishing'].hist(bins=100, figsize=(20, 5))


# In[ ]:


for c in pred_cols:
    print(c)
    display((df_test[c] * 10).astype(int).value_counts(sort=False))


# In[ ]:


# 0     1504
# 1       11
# 2        6
# 3        6
# 4        2
# 5        3
# 6        2
# 9       39
# 10      53
(df_test['smishing'] * 10).astype(int).value_counts(sort=False)


# In[ ]:


model_ts


# In[ ]:


df_test[['smishing']].to_csv('submit/{}_nn.csv'.format(model_ts), index=True)
# df_test[['id', 'smishing', 'text']].sort_values('smishing', ascending=False).to_csv('{}_text.csv'.format(model_ts), index=False)


# In[ ]:





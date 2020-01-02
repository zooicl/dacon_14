
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

print(os.getcwd())

base_path = '.'

df_train = pd.read_csv(os.path.join(base_path , 'input/train.csv'), index_col=0)
df_test = pd.read_csv(os.path.join(base_path , 'input/public_test.csv'), index_col=0)
df_test['smishing'] = -1

df_fea = pd.concat([df_train, df_test])
df_fea.shape


train_size = len(df_train)

### Mecab

# mecab = Mecab()
# # df_space['morphs'] = df_space['spacing'].apply(lambda x: mecab.morphs(x))
# df_fea['nouns'] = df_fea['text'].apply(lambda x: mecab.nouns(x))

# df_fea['nouns_str'] = df_fea['nouns'].apply(lambda x: ' '.join(x))

# df_fea.to_pickle('df_fea.pkl')

df_fea = pd.read_pickle('data/df_fea.pkl')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_data = df_fea[:train_size]['nouns_str'].values
y_data = df_fea['smishing'].values

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) 
sequences = tokenizer.texts_to_sequences(X_data)

X_data

word_to_index = tokenizer.word_index
print(word_to_index)

vocab_size = len(word_to_index) + 1
print('단어 집합의 크기: {}'.format((vocab_size)))

n_of_train = int(train_size * 0.8)
n_of_test = int(train_size - n_of_train)
print(n_of_train)
print(n_of_test)

X_data = sequences
max_len = max(len(l) for l in X_data)
print('메일의 최대 길이 : %d' % max(len(l) for l in X_data))
print('메일의 평균 길이 : %f' % (sum(map(len, X_data))/len(X_data)))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of Data')
plt.ylabel('number of Data')
plt.show()

data = pad_sequences(X_data, maxlen=max_len)
print("data shape: ", data.shape)

X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
y_test = np.array(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
y_train = np.array(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 4457개의 데이터만 저장

### RNN

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

import tensorflow as tf
with tf.device('/device:XLA_GPU:0'):
    model = Sequential()
    model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
    model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, epochs=4, batch_size=64, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
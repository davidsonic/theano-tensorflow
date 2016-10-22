from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Embedding
from keras.layers import LSTM,SimpleRNN,GRU
from keras.datasets import imdb

max_features=10
maxlen=80
batch_size=32

print('loading data...')
(X_train,y_train),(X_test,y_test)=imdb.load_data(nb_words=max_features)
print(len(X_train),'train sequences')
print(len(X_test),'test sequences')
print(X_train.shape)
print(y_train.shape)


print('Pad sequences (samples x time)')
X_train=sequence.pad_sequences(X_train,maxlen=maxlen)
X_test=sequence.pad_sequences(X_test,maxlen=maxlen)
print('X_train shape:',X_train.shape)
print('X_test shape:', X_test.shape)


print('Build model...')
model=Sequential()
model.add(Embedding(max_features,128,dropout=0.2))
model.add(LSTM(128,dropout_W=0.2,dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train,y_train,batch_size=batch_size,nb_epoch=15,
          validation_data=(X_test,y_test))

score,acc=model.evaluate(X_test,y_test,batch_size=batch_size)

print('Test score:',score)
print('Test accuracy:',acc)

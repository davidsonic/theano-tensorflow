from keras.models import Sequential
from keras.layers import LSTM, Dense

data_dim=16
timesteps=8
nb_classes=10
batch_size=32

model=Sequential()
model.add(LSTM(32,return_sequences=True,stateful=True,
               batch_input_shape=(batch_size,timesteps,data_dim)))
model.add(LSTM(32,return_sequences=True,stateful=True))

model.add(LSTM(32,stateful=True))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

import numpy as np
x_train=np.random.random((batch_size*10,timesteps,data_dim))
y_train=np.random.random((batch_size*10,nb_classes))

x_val=np.random.random((batch_size*3,timesteps,data_dim))
y_val=np.random.random((batch_size*3,nb_classes))

model.fit(x_train,y_train,
          batch_size=batch_size,nb_epoch=5,
          validation_data=(x_val,y_val))




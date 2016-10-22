from keras.models import Sequential
from keras.layers import Dense

model=Sequential()
model.add(Dense(1,input_dim=784,activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

import numpy as np
data=np.random.random((1000,784))
labels=np.random.randint(2,size=(1000,1))

model.fit(data,labels,nb_epoch=10,batch_size=32)

# for a multi-input model with 10 classes

from theano3 import train
import pickle
from theano3 import *


'''
save model
'''

#save model

for i in range(500):
    train(D[0],D[1])


with open('save/model1.pickle','wb') as file:
    model=[W.get_value(),b.get_value()]
    pickle.dump(model,file)
    print (model[0][:10])
    print('accuracy: ',compute_accuracy(D[1],predict(D[0])))


#load model

with open('save/model1.pickle','rb') as file:
    model=pickle.load(file)
    W.set_value(model[0])
    b.set_value(model[1])
    print (W.get_value()[:10])
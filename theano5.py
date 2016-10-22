import numpy as np
from theano4 import Layer
import theano.tensor as T
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import theano

'''
overfitting  regularization
'''

#regularization l1,l2

def minmax_normalization(data):
    xs_max=np.max(data,axis=0)
    xs_min=np.min(data,axis=0)
    xs=(1-0)*(data-xs_min)/(xs_max-xs_min)+0
    return xs


np.random.seed(100)
x_data=load_boston().data
x_data=minmax_normalization(x_data)
y_data=load_boston().target[:,np.newaxis] #add a dimension to become matrix,corresponds with y=T.dmatrix('y')


#cross validataion, train test data split
x_train,y_train=x_data[:400],y_data[:400]
x_test,y_test=x_data[400:],y_data[400:]

x=T.dmatrix('x')
y=T.dmatrix('y')

l1=Layer(x,13,50,T.tanh)   #13 equals x_data variable
l2=Layer(l1.outputs,50,1,None)

# cost=T.mean(T.square(l2.outputs-y))
cost=T.mean(T.square(l2.outputs-y))+0.1*((l1.W**2).sum()+(l2.W**2).sum())   #l2 regularization
# cost=T.mean(T.square(l2.outputs-y))+0.1*(abs(l1.W).sum()+abs(l2.W).sum())   #l1 regularziation
gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])

learning_rate=0.01
train=theano.function(inputs=[x,y],updates=[(l1.W,l1.W-learning_rate*gW1),
                                            (l1.b,l1.b-learning_rate*gb1),
                                            (l2.W,l2.W-learning_rate*gW2),
                                            (l2.b,l2.b-learning_rate*gb2)])

compute_cost=theano.function(inputs=[x,y],outputs=cost)

#record cost
train_err_list=[]
test_err_list=[]
learning_time=[]

for i in range(1000):
    train(x_train,y_train)
    if i%10==0:
        train_err_list.append(compute_cost(x_train,y_train))
        test_err_list.append(compute_cost(x_test,y_test))
        learning_time.append(i)


# plot cost history
plt.plot(learning_time,train_err_list,'r-')
plt.plot(learning_time,test_err_list,'b--')
plt.show()



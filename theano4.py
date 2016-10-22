import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt



'''
regression
'''
# define layer
'''
l1=Layer(inputs,in_size=1,out_size=10,activation_function)
l2=Layer(l1.outputs,10,out_size)
'''

class Layer(object):
    def __init__(self,inputs,in_size,out_size,activation_function=None):
        self.W=theano.shared(np.random.normal(0,1,(in_size,out_size)))
        self.b=theano.shared(np.zeros((out_size,))+0.1)
        self.Wx_plus_b=T.dot(inputs,self.W)+self.b
        self.activation_function=activation_function
        if activation_function is None:
            self.outputs=self.Wx_plus_b
        else:
            self.outputs=self.activation_function(self.Wx_plus_b)


#make up some fake data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
noise=np.random.normal(0,0.05,x_data.shape)
y_data=np.square(x_data)-0.5+noise

#show the fake data
# plt.scatter(x_data,y_data)
# plt.show()


# determine the inputs dtype
x=T.dmatrix('x')
y=T.dmatrix('y')

#add layers
l1=Layer(x,1,10,T.nnet.relu)   #input_size=1 because x_data is one dimensional
l2=Layer(l1.outputs,10,1,None)


# compute the cost
cost=T.mean(T.square(l2.outputs-y))  # get mean error of 300 samples


#compute the gradients
gW1,gb1,gW2,gb2=T.grad(cost,[l1.W,l1.b,l2.W,l2.b])


#apply gradient descent
learning_rate=0.05
train=theano.function(inputs=[x,y],outputs=cost,updates=[(l1.W,l1.W-learning_rate*gW1),
                                                        (l1.b,l1.b-learning_rate*gb1),
                                                        (l2.W,l2.W-learning_rate*gW2),
                                                        (l2.b,l2.b-learning_rate*gb2)]
                      )

#predictions
predict=theano.function(inputs=[x],outputs=l2.outputs)


#plot the fake data
fig=plt.figure()
ax=fig.add_subplot(1,1,1)   #number one
ax.scatter(x_data,y_data)
plt.ion()  # not block
plt.show()  #block is default value, plt.show(block=True)

#below is true application:
for i in range(1000):
    #training
    err=train(x_data,y_data)  #put real data
    if i%50==0:
        print(err)
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #to visualize the result and improvement
        predicted_value=predict(x_data)
        #plot the prediction
        lines=ax.plot(x_data,predicted_value,'r-',lw=5)
        plt.pause(1)  # pause for one second


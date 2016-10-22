'''
overfitting/drop out
This is the second about tensorboard 2
'''

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3)


def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    Wx_plus_b=tf.nn.dropout(Wx_plus_b,keep_prob)  #for drop out
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b,)
    tf.histogram_summary(layer_name+'/outputs',outputs)
    return outputs


# define placeholder
keep_prob=tf.placeholder(tf.float32)  #for drop out
xs=tf.placeholder(tf.float32,[None,64])  # not regulate how many samples, but each sample has 8*8=64 pixels
ys=tf.placeholder(tf.float32,[None,10])

#add output layer
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)  #100 to overfit, tanh avoid NaN
prediction=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)


# loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]),
                             )  # refer to math_ops.py for its meaning
tf.scalar_summary('loss',cross_entropy) #tensorflow has to have at least one histogram

train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess=tf.Session()
merged=tf.merge_all_summaries()
# summary writer
train_writer=tf.train.SummaryWriter('logs/train',sess.graph)
test_writer=tf.train.SummaryWriter('logs/test',sess.graph)

sess.run(tf.initialize_all_variables())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5}) #keep probablity,retain 50%, test loss and train loss becomes closer
    if i%50==0:
        #record loss
        train_result=sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result=sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)

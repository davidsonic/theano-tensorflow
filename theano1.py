import theano.tensor as T
from theano import function
import theano
import  numpy as np

#scalar
x=T.dscalar('x')
y=T.dscalar('y')
z=x+y
f=function([x,y],z)
print(f(2,3))

#matrix
x=T.dmatrix('x')
y=T.dmatrix('y')
z=x+y
f=function([x,y],z)

print(f(np.arange(12).reshape(3,4),
      10*np.ones((3,4))))


#activation
x=T.dmatrix('x')
s=1/(1+T.exp(-x))
logistic=theano.function([x],s)
print(logistic([[0,-1],[-2,-3]]))


#multiple outputs
a,b=T.dmatrices('a','b')
diff=a-b
abs_diff=abs(diff)
diff_squared=diff**2
f=theano.function([a,b],[diff,abs_diff,
                         diff_squared])

x1,x2,x3=f(np.ones((2,2)),np.arange(4).reshape((2,2)))

print x1
print x2
print x3
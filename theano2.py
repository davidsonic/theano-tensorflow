import theano.tensor as T
import theano
import numpy as np


#name for a function
x,y,w=T.dscalars('x','y','z')
z=(x+y)*w
f=theano.function([x,theano.In(y,value=1),
                   theano.In(w,value=2,name='weights')],
                  z)

print(f(23,2,weights=4))


#shared value
state=theano.shared(np.array(0,dtype=np.float64),
                    'state')
inc=T.scalar('inc',dtype=state.dtype)
accumulator=theano.function([inc],state,
                            updates=[(state,state+inc)])

#get variable
print(state.get_value())
accumulator(1)
print(state.get_value())
accumulator(10)
print(state.get_value())


#set value
state.set_value(-1)
accumulator(3)
print(state.get_value())

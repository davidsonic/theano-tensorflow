import numpy as np

vocab_size=10
input_size=vocab_size
hidden_size=100
Wxh=np.zeros((hidden_size,input_size))
Whh=np.zeros((hidden_size,hidden_size))
bh=np.zeros((hidden_size,1))
Why=np.zeros((vocab_size,hidden_size))


def sample(h,seed_ix,n):
    '''
    sample a sequence of n integers from the model
    h is the memory state, seed_ix is seed letter for first time step
    :param h:
    :param seed_ix:
    :param n:
    :return:
    '''

    x=np.zeros((vocab_size,1))
    x[seed_ix]=1

    ixes=[]

    for t in xrange(n):
        h=np.tanh(np.dot(Wxh,x)+np.dot(Whh,h)+bh)
        y=np.dot(Why,h)+bh
        p=np.exp(y)/np.sum(np.exph(y))
        ix=np.random.choice(range(vocab_size),p=p.ravel())

        x=np.zeros((vocab_size,1))
        x[ix]=1

        ixes.append(ix)

    return ixes


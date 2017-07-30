from . import trace
from . import plot
from . import mnist
from . import tuning

def get_ae_type(directory):
    import os.path
    d, n = os.path.split(directory)
    if n == '':
        return d.split("/")[-1].split("_")[-1]
    else:
        return n.split("_")[-1]
    

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def prepare_binary_classification_data(real, fake):
    import numpy as np
    shape = real.shape[1:]
    
    both = np.concatenate((real, fake),axis=0)
    both = np.reshape(both, (len(both), -1)) # flatten

    data_dim = both.shape[1]

    both2 = np.pad(both, ((0,0),(0,1)), 'constant') # default 0
    both2[:len(real),-1] = 1                        # tag true
    np.random.shuffle(both2)

    train_in  = np.reshape(both2[:int(0.9*len(both2)), :data_dim], (-1, *shape))
    train_out = both2[:int(0.9*len(both2)), -1]
    test_in   = np.reshape(both2[int(0.9*len(both2)):, :data_dim], (-1, *shape))
    test_out  = both2[int(0.9*len(both2)):, -1]

    return train_in, train_out, test_in, test_out


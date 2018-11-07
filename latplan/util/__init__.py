from . import trace
from . import plot
from . import mnist
from . import tuning
from . import timer
from . import noise

def get_ae_type(directory):
    import os.path
    d, n = os.path.split(directory)
    if n == '':
        return d.split("/")[-1].split("_")[1]
    else:
        return n.split("_")[1]

def ensure_directory(directory):
    if directory[-1] is "/":
        return directory
    else:
        return directory+"/"
    

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


import numpy as np
def set_difference(a, b):
    assert a.shape[1:] == b.shape[1:]
    a = a.copy()
    b = b.copy()
    a_v = a.view([('', a.dtype)] * a.shape[1])
    b_v = b.view([('', b.dtype)] * b.shape[1])
    return np.setdiff1d(a_v, b_v).view(a.dtype).reshape((-1, a.shape[1]))

def union(a, b):
    assert a.shape[1:] == b.shape[1:]
    d = set_difference(a, b)
    return np.concatenate((b, d), axis=0)

 
def bce(true,pred,axis=None,epsilon=1e-7):
    x = true
    y = pred
    return - (x * np.log(np.clip(y,epsilon,1)) + \
              (1-x) * np.log(np.clip(1-y,epsilon,1))).mean(axis=axis)

def mae(x,y,axis=None):
    return np.mean(np.absolute(x - y),axis=axis)


def mse(x,y,axis=None):
    return np.sqrt(np.sum(np.square(x - y),axis=axis))


def puzzle_module(directory):
    import importlib
    from latplan.util import ensure_directory
    args = ensure_directory(directory).split("/")[-2].split("_")
    if args[0] == "hanoi":
        p = importlib.import_module('latplan.puzzles.hanoi')
    else:
        p = importlib.import_module('latplan.puzzles.{}_{}'.format(args[0],args[2]))
    p.setup()
    return p

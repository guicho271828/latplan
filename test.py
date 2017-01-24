#!/usr/bin/env python3

import config
import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

from plot import plot_ae, plot_grid

if __name__ == '__main__':
    import numpy.random as random
    from trace import trace
    
    import mnist_puzzle
    configs1 = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                [1, 0, 2, 3, 4, 5, 6, 7, 8],  # right
                [3, 1, 2, 0, 4, 5, 6, 7, 8],  # down
                [2, 1, 0, 3, 4, 5, 6, 7, 8],] # invalid
    configs2 = [[0, 1, 2, 3, 8, 7, 6, 5, 4],
                [1, 0, 2, 3, 8, 7, 6, 5, 4],
                [3, 1, 2, 0, 8, 7, 6, 5, 4],
                [2, 1, 0, 3, 8, 7, 6, 5, 4],]
    
    x1       = mnist_puzzle.states(3,3,configs1)
    x2       = mnist_puzzle.states(3,3,configs2)
    
    ae = GumbelAE("samples/mnist_puzzle33p_model/").load()

    b1 = ae.encode_binary(x1).round().astype(bool)
    b2 = ae.encode_binary(x2).round().astype(bool)

    o1 = b1[0]
    o2 = b2[0]

    diff1 = np.array([ np.bitwise_xor(o1,b) for b in b1 ])
    diff2 = np.array([ np.bitwise_xor(o2,b) for b in b2 ])

    plot_ae(ae,x1,"image1.png")
    plot_ae(ae,x2,"image2.png")
    plot_grid(np.array([b1,diff1,diff2,b2]).reshape((-1,5,5)),w=4,path=ae.local("diff.png"))

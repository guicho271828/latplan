#!/usr/bin/env python3

import config
import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE, Discriminator

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################


from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

if __name__ == '__main__':
    import numpy.random as random
    from trace import trace
    import mnist_puzzle
    configs = mnist_puzzle.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = mnist_puzzle.states(3,3,train_c)
    test        = mnist_puzzle.states(3,3,test_c)
    train_random     = random.rand(*train.shape)
    test_random      = random.rand(*test.shape)

    d1 = np.concatenate((train, train_random), axis=0)
    d2 = np.concatenate((test,  test_random),  axis=0)
    l1 = np.concatenate((np.ones(len(train)), np.zeros(len(train_random))), axis=0)
    l2 = np.concatenate((np.ones(len(test)),  np.zeros(len(test_random))), axis=0)
    raw_discriminator = Discriminator("samples/mnist_puzzle33p_raw_discriminator/",{0:2000,1:0.4}) 
    raw_discriminator.train(d1, test_data=d2, train_data_to=l1, test_data_to=l2)

    ae = GumbelAE("samples/mnist_puzzle33p_model/").load()
    b1 = ae.encode_binary(d1)
    b2 = ae.encode_binary(d2)
    bin_discriminator = Discriminator("samples/mnist_puzzle33p_bin_discriminator/",{0:2000,1:0.4})
    bin_discriminator.train(b1, test_data=b2, train_data_to=l1, test_data_to=l2)
    
    
    
    

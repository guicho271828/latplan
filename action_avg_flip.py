#!/usr/bin/env python3
import warnings
import config
import numpy as np
import model

import keras.backend as K

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

GAN = model.ActionGan

np.set_printoptions(threshold=np.inf)


if __name__ == '__main__':
    import numpy.random as random
    from trace import trace

    data = np.loadtxt("samples/mnist_puzzle33_fc2/all_actions.csv",dtype=np.int8)
    
    pre, suc = data[:,:49], data[:,49:]

    abs_diff = np.sum(np.abs(pre-suc),axis=1)

    hist = np.histogram(abs_diff,49,(0,49))

    print(hist)


    # 13 bit flips at most

#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

import mnist_puzzle
states = mnist_puzzle.states(3,2)

import numpy.random as random
def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def run(ae,xs):
    zs = ae.encode_binary(xs)
    ys = ae.decode_binary(zs)
    mod_ys = []
    correlations = []
    print(ys.shape)
    print("corrlations:")
    print("bit \ image  {}".format(range(len(xs))))
    for i in range(ae.N):
        mod_zs = np.copy(zs)
        # increase the latent value from 0 to 1 and check the difference
        for j in range(11):
            mod_zs[:,i] = j / 10.0
            mod_ys.append(ae.decode_binary(mod_zs))
        zero_zs,one_zs = np.copy(zs),np.copy(zs)
        zero_zs[:,i] = 0.
        one_zs[:,i] = 1.
        correlation = np.mean(np.square(ae.decode_binary(zero_zs) - ae.decode_binary(one_zs)),
                              axis=(1,2))
        correlations.append(correlation)
        print("{}            {}".format(i,correlation))
    plot_grid2(np.einsum("ib...->bi...",np.array(mod_ys)).reshape((-1,)+ys.shape[1:]),
               w=11,path=ae.local("dump_significance.png"))
    return np.einsum("ib->bi",correlations)

# run(GumbelAE("samples/mnist_puzzle32_model/"),states[0:1])
run(GumbelAE("samples/mnist_puzzle32p_model/"),select(states,6))

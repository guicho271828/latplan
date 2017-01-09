#!/usr/bin/env python

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

def run(ae):
    m = 16
    zs = (((np.arange(2**m)[:,None] & (1 << np.arange(m)))) > 0).astype(int)
    ys = ae.decode_binary(zs)
    per_image = 2**8
    for j in range((2**16) // per_image):
        path = ae.local("all-bits{}.png".format(j))
        print path
        plot_grid2(ys[j*per_image:(1+j)*per_image],w=16,path=path)

run(GumbelAE("samples/mnist_puzzle32_model/"))
run(GumbelAE("samples/mnist_puzzle32p_model/"))

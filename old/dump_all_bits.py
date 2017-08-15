#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, GumbelAE2, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

def run(ae):
    ae.load()
    n = ae.parameters['N']
    zs = (((np.arange(2**n)[:,None] & (1 << np.arange(n)))) > 0).astype(int)
    ys = ae.decode_binary(zs)
    per_image = 2**8
    for j in range((2**n) // per_image):
        path = ae.local("all-bits{}.png".format(j))
        print(path)
        plot_grid2(ys[j*per_image:(1+j)*per_image],w=n,path=path)

# run(GumbelAE2("samples/lightsout_digital_4_fc2"))
# run(GumbelAE( "samples/lightsout_twisted_3_fc"))
# run(GumbelAE2("samples/hanoi3_fc2"))
# run(GumbelAE2("samples/hanoi4_fc2"))
# run(GumbelAE2("samples/mandrill_puzzle33_fc2"))
# run(GumbelAE2("samples/mnist_puzzle33_fc2"))
# run(GumbelAE2("samples/xhanoi4_fc2"))



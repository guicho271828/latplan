#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, GumbelAE2, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

def run(ae):
    ae.load()
    n = ae.parameters['N']
    zs = np.identity(n,int)
    ys = ae.decode_binary(zs)
    path = ae.local("each-bit.png")
    plot_grid(ys,w=1,path=path)
    zs = 1-np.identity(n,int)
    ys = ae.decode_binary(zs)
    path = ae.local("each-bit-neg.png")
    plot_grid(ys,w=1,path=path)

run(GumbelAE2("samples/lightsout_digital_4_fc2"))
run(GumbelAE( "samples/lightsout_twisted_3_fc"))
run(GumbelAE2("samples/hanoi3_fc2"))
run(GumbelAE2("samples/hanoi4_fc2"))
run(GumbelAE2("samples/mandrill_puzzle33_fc2"))
run(GumbelAE2("samples/mnist_puzzle33_fc2"))
run(GumbelAE2("samples/xhanoi4_fc2"))

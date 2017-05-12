#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

def run(ae):
    m = ae.parameters['M']
    zs = (((np.arange(2**m)[:,None] & (1 << np.arange(m)))) > 0).astype(int)
    ys = ae.decode_binary(zs)
    per_image = 2**8
    for j in range((2**m) // per_image):
        path = ae.local("all-bits{}.png".format(j))
        print(path)
        plot_grid2(ys[j*per_image:(1+j)*per_image],w=m,path=path)

run(GumbelAE("samples/digital_lightsout_4_fc2/"))


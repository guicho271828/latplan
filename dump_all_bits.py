#!/usr/bin/env python

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
from plot import plot_grid, plot_grid2

ae = GumbelAE("samples/puzzle32p_model/")
# ae = ConvolutionalGumbelAE("samples/puzzle32p_modelc/")
ae.load()

m = 16

zs = (((np.arange(2**m)[:,None] & (1 << np.arange(m)))) > 0).astype(int)
ys = ae.decode_binary(zs)

def run():
    per_image = 2**8
    for j in range((2**16) // per_image):
        # print "gathering..."
        # images = []
        # for z,y in zip(zs[j*per_image:(1+j)*per_image],
        #                ys[j*per_image:(1+j)*per_image]):
        #     images.append(z.reshape((4,4)))
        #     images.append(y)
        path = ae.local("all-bits{}.png".format(j))
        print path
        # plot_grid(images,path)
        # plot_grid(ys[j*per_image:(1+j)*per_image],path)
        plot_grid2(ys[j*per_image:(1+j)*per_image],shape=(12,15),w=16,path=path)

run()

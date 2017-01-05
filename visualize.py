#!/usr/bin/env python

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
# from quiver_engine import server

# ae = GumbelAE("samples/puzzle22_model/")
ae = ConvolutionalGumbelAE("samples/puzzle22_modelc/")
ae.load()
# server.launch(ae.encoder)

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, GaussianNoise
from keras.models import Model, Sequential

conv_layers = [ l for l in ae.encoder.layers if isinstance(l,Convolution2D) ]

from plot import plot_grid, plot_grid2

def print_conv():
    for i,conv in enumerate(conv_layers):
        W = conv.get_weights()[0]
        path = ae.local("conv_filters{}.png".format(i))
        print path
        plot_grid(np.einsum("xycf->fxy",W),path=path)

print_conv()

import puzzle
xs = puzzle.states(2,2)
import numpy.random as random
xs = xs[random.randint(0,xs.shape[0],12)]

_x = ae.encoder.layers[0]

def print_activation():
    path = ae.local("activation{}.png".format(0))
    print path, xs.shape
    plot_grid(xs,path=path)
    x = Input(_x.input_shape[1:])
    now = x
    for i,l in enumerate(ae.encoder.layers[1:]):
        now = l(now)
        if isinstance(l,Convolution2D):
            m = Model(x,now)
            now_images = m.predict(xs)
            path = ae.local("activation{}.png".format(i+1))
            print path
            plot_grid(now_images,path=path)

print_activation()

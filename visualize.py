#!/usr/bin/env python3

import numpy as np
from model import GumbelAE, ConvolutionalGumbelAE
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, GaussianNoise
from keras.models import Model, Sequential
# from quiver_engine import server
# server.launch(ae.encoder)
import puzzle
import numpy.random as random

from plot import plot_grid, plot_grid2

def print_conv(ae):
    conv_layers = [ l for l in ae.encoder.layers if isinstance(l,Convolution2D) ]
    for i,conv in enumerate(conv_layers):
        W = conv.get_weights()[0]
        path = ae.local("conv_filters{}.png".format(i))
        print(path)
        plot_grid(np.einsum("xycf->fxy",W),path=path)

def print_activation(xs,shape):
    path = ae.local("activation{}.png".format(0))
    print(path, xs.shape)
    plot_grid(xs,path=path)
    x = Input(shape)
    now = x
    for i,l in enumerate(ae.encoder.layers[1:]):
        now = l(now)
        if isinstance(l,Convolution2D):
            m = Model(x,now)
            now_images = m.predict(xs)
            path = ae.local("activation{}.png".format(i+1))
            print(path)
            plot_grid(now_images,path=path)

# ae = GumbelAE("samples/puzzle32_model/")
ae = ConvolutionalGumbelAE("samples/puzzle32p_modelc/")
ae.load()

print_conv(ae)

xs = puzzle.states(3,2)
xs = xs[random.randint(0,xs.shape[0],12)]
print_activation(xs,ae.encoder.layers[0].input_shape[1:])

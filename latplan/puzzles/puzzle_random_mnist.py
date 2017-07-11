#!/usr/bin/env python3

import numpy as np
from .model.puzzle import generate_configs, successors

from ..util.mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]
from numpy.random import randint

def random_panels():
    return [ digits[randint(0,len(digits))].reshape((28,28)) for digits in imgs ]

base_width = 28
base_height = 28
def generate(configs, width, height, panels):
    assert width*height <= 9
    dim_x = base_width*width
    dim_y = base_height*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*base_height:(y+1)*base_height,
                   x*base_width:(x+1)*base_width] = panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

def states(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return np.array([ generate([c],width,height,random_panels()) for c in configs ]).reshape([-1,base_height*height,base_width*width])

def transitions(width, height, configs=None, one_per_state=False):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        def pickone(thing):
            index = randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate(
                [c1,pickone(successors(c1,width,height))],width,height,random_panels())
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],width,height,random_panels())
                                 for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)


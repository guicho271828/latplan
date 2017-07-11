#!/usr/bin/env python3

import numpy as np
from .model.counter import generate_configs, successors
from numpy.random import randint
from ..util.mnist import mnist

x_train, y_train, filters, imgs = None, None, None, None

def load():
    global x_train, y_train, filters, imgs
    if x_train is None:
        x_train, y_train, _, _ = mnist()
        filters = [ np.equal(i,y_train) for i in range(10) ]
        imgs    = [ x_train[f] for f in filters ]

def random_panels():
    load()
    return [ digits[randint(0,len(digits))].reshape((28,28)) for digits in imgs ]

def generate(configs,panels):
    load()
    return np.array([ panels[c] for c in configs ])

def states(size,configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs,random_panels())

def transitions(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    transitions = np.array([ generate([c1,c2],random_panels())
                             for c1 in configs for c2 in successors(c1,size) ])
    return np.einsum('ab...->ba...',transitions)



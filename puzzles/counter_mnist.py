#!/usr/bin/env python3

import numpy as np
from .model.counter import generate_configs, successors

from ..util.mnist import mnist

x_train, y_train, filters, imgs, panels = None, None, None, None, None

def load():
    global x_train, y_train, filters, imgs, panels
    if x_train is None:
        x_train, y_train, _, _ = mnist()
        filters = [ np.equal(i,y_train) for i in range(10) ]
        imgs    = [ x_train[f] for f in filters ]
        panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]

def generate(configs):
    load()
    return np.array([ panels[c] for c in configs ])

def states(size,configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    transitions = np.array([ generate([c1,c2])
                             for c1 in configs for c2 in successors(c1,size) ])
    return np.einsum('ab...->ba...',transitions)



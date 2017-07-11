#!/usr/bin/env python3

import numpy as np
from .model.lightsout import generate_configs, generate_random_configs, successors

on = [[0, 0, 0, 0, 0,],
      [0, 0, 1, 0, 0,],
      [0, 1, 1, 1, 0,],
      [0, 0, 1, 0, 0,],
      [0, 0, 0, 0, 0,],]

off= [[0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0,],
      [0, 0, 0, 0, 0,],]

def generate(configs):
    import math
    size = int(math.sqrt(len(configs[0])))
    base = 5
    dim = base*size
    def generate(config):
        figure = np.zeros((dim,dim))
        for pos,value in enumerate(config):
            x = pos % size
            y = pos // size
            if value > 0:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = on
            else:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = off
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim,dim))

def states(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None, one_per_state=False):
    if configs is None:
        configs = generate_configs(size)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate([c1,pickone(successors(c1))])
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2])
                                 for c1 in configs for c2 in successors(c1) ])
    return np.einsum('ab...->ba...',transitions)


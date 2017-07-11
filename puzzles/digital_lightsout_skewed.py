#!/usr/bin/env python3

import numpy as np
from .model.lightsout import generate_configs, generate_random_configs, successors

on = [[0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 1, 1, 1, 1, 1, 1, 1, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],]
                            
off= [[0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, ],]

def generate(configs):
    import math
    size = int(math.sqrt(len(configs[0])))
    base = 9
    dim = base*size
    half = dim//2
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
        # np.skew or np.XXX or any visual effect
        figure2 = np.zeros((dim*4,dim*4))
        figure2[dim+half:2*dim+half,
                dim+half:2*dim+half] = figure
        figure3 = np.zeros((dim*2,dim*2))
        for x in range(-dim,dim):
            for y in range(-dim,dim):
                r = math.sqrt(x*x+y*y)
                rad = math.atan2(y,x) + math.pi  * (1-r)/half/5
                p = r * np.array([math.cos(rad), math.sin(rad)])
                px1 = math.floor(p[0])
                py1 = math.floor(p[1])
                grid = np.array([[[px1,py1],[px1+1,py1],],
                                 [[px1,py1+1],[px1+1,py1+1],],])
                w = (1 - np.prod(np.fabs(grid - p),axis=-1))/3
                
                value = np.sum(w * figure2[py1+2*dim:py1+2+2*dim,px1+2*dim:px1+2+2*dim])
                figure3[y+dim,x+dim] = value*2
        return figure3
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim*2,dim*2)).clip(0,1)

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



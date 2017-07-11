#!/usr/bin/env python3

import numpy as np
from .model.puzzle import generate_configs, successors

def generate(configs, width, height):
    assert width*height <= 16
    base_width = 5
    base_height = 6
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
    return generate(configs,width,height)

def transitions(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    transitions = np.array([ generate([c1,c2],width,height)
                             for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)

panels = [
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 0, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 0, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 1, 0,],
     [0, 1, 0, 1, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 0, 1, 0,],
     [0, 1, 1, 1, 0,],
     [0, 1, 0, 0, 0,],
     [0, 0, 1, 1, 0,],],
    [[0, 0, 0, 0, 0,],
     [0, 0, 0, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 1, 1, 1, 0,],
     [0, 0, 1, 0, 0,],
     [0, 0, 1, 0, 0,],],
]



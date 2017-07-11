#!/usr/bin/env python3

import os
import numpy as np

from .model.puzzle import generate_configs, successors
from .split_image import split_image

panels = None

base = 21

def generate(configs, width, height):
    global panels
    if panels is None:
        panels  = split_image(os.path.join(os.path.dirname(__file__), "lenna.png"),width,height)
        stepy = panels[0].shape[0]//base
        stepx = panels[0].shape[1]//base
        panels = panels[:,::stepy,::stepx][:,:base,:base].round()
    assert width*height <= 9
    dim_x = base*width
    dim_y = base*height
    def generate(config):
        figure = np.zeros((dim_y+height-1,dim_x+width-1))
        for pos,digit in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*(base+1):(y+1)*(base+1)-1,
                   x*(base+1):(x+1)*(base+1)-1] = panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y+height-1,dim_x+width-1))

def states(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height)

def transitions(width, height, configs=None, one_per_state=False):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate(
                [c1,pickone(successors(c1,width,height))],width,height)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],width,height)
                                 for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)


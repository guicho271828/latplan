#!/usr/bin/env python3

import numpy as np
from .model.puzzle import setting, generate, states, transitions, generate_configs, successors
from .model.puzzle import validate_states, validate_transitions, to_configs
from .split_image import split_image
import os

def setup():
    setting['base'] = 21

    def loader(width,height):
        base = setting['base']
        panels = split_image(os.path.join(os.path.dirname(__file__), "mandrill.bmp"),width,height)
        stepy = panels[0].shape[0]//base
        stepx = panels[0].shape[1]//base
        return panels[:,::stepy,::stepx][:,:base,:base].round()

    setting['loader'] = loader

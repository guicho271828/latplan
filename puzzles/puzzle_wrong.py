#!/usr/bin/env python3

import numpy as np
from .model.puzzle import *
from .split_image import split_image
from .util import preprocess
import os

def setup():
    setting['base'] = 14

    def loader(width,height):
        from ..util.mnist import mnist
        base = setting['base']
        x_train, y_train, _, _ = mnist()
        filters = [ np.equal(i,y_train) for i in range(9) ]
        imgs    = [ x_train[f] for f in filters ]
        panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]
        panels[8] = imgs[8][3].reshape((28,28))
        panels[1] = imgs[8][3].reshape((28,28))
        panels = np.array(panels)
        stepy = panels.shape[1]//base
        stepx = panels.shape[2]//base
        # unfortunately the method below generates "bolder" fonts
        # panels = panels[:,:stepy*base,:stepx*base,]
        # panels = panels.reshape((panels.shape[0],base,stepy,base,stepx))
        # panels = panels.mean(axis=(2,4))
        # panels = panels.round()
        panels = panels[:,::stepy,::stepx][:,:base,:base].round()
        panels = preprocess(panels)
        return panels

    setting['loader'] = loader


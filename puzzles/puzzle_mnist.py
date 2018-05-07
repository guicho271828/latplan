#!/usr/bin/env python3

import numpy as np
from .model.puzzle import setting, generate, states, transitions, generate_configs, successors
from .model.puzzle import validate_states, validate_transitions, to_configs, to_objects
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
        panels[1] = imgs[1][3].reshape((28,28))
        # additional panels
        # panels.append(np.random.uniform(0,1,(28,28)))
        # panels.append(np.zeros((28,28)))
        # panels.append(np.ones((28,28))*255)
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


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     def plot_image(a,name):
#         plt.figure(figsize=(6,6))
#         plt.imshow(a,interpolation='nearest',cmap='gray',)
#         plt.savefig(name)
#     
#     def plot_grid(images,name="plan.png"):
#         import matplotlib.pyplot as plt
#         l = len(images)
#         w = 6
#         h = max(l//6,1)
#         plt.figure(figsize=(20, h*2))
#         for i,image in enumerate(images):
#             # display original
#             ax = plt.subplot(h,w,i+1)
#             plt.imshow(image,interpolation='nearest',cmap='gray',)
#             ax.get_xaxis().set_visible(False)
#             ax.get_yaxis().set_visible(False)
#         plt.savefig(name)
#     
#     configs = list(generate_configs(9))[:36]
#     puzzles = generate(configs, 3, 3)
#     plot_grid(puzzles,"mnist_puzzles.png")
#     _transitions = transitions(3,3,configs)
#     import numpy.random as random
#     indices = random.randint(0,_transitions[0].shape[0],18)
#     _transitions = _transitions[:,indices]
#     print(_transitions.shape)
#     transitions_for_show = \
#         np.einsum('ba...->ab...',_transitions) \
#           .reshape((-1,)+_transitions.shape[2:])
#     print(transitions_for_show.shape)
#     plot_grid(transitions_for_show,"mnist_puzzle_transitions.png")
#     
#     dummy = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
#              [9, 9, 9, 9, 9, 9, 9, 9, 9],
#              [10, 10, 10, 10, 10, 10, 10, 10, 10],
#              [11, 11, 11, 11, 11, 11, 11, 11, 11],
#              [12, 12, 12, 12, 12, 12, 12, 12, 12]]
#     plot_grid(generate(dummy, 3, 3),"mnist_puzzles_dummy.png")


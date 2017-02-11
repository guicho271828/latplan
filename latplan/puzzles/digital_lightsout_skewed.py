#!/usr/bin/env python3

import numpy as np
from .lightsout import generate_configs, successors

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
        # np.skew or np.XXX or any visual effect
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim,dim))

def states(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    transitions = np.array([ generate([c1,c2])
                             for c1 in configs for c2 in successors(c1) ])
    return np.einsum('ab...->ba...',transitions)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def plot_image(a,name):
        plt.figure(figsize=(6,6))
        plt.imshow(a,interpolation='nearest',cmap='gray',)
        plt.savefig(name)
    def plot_grid(images,name="plan.png"):
        import matplotlib.pyplot as plt
        l = len(images)
        w = 6
        h = max(l//6,1)
        plt.figure(figsize=(20, h*2))
        for i,image in enumerate(images):
            # display original
            ax = plt.subplot(h,w,i+1)
            plt.imshow(image,interpolation='nearest',cmap='gray',)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(name)
    configs = generate_configs(3)
    puzzles = generate(configs)
    print(puzzles[10])
    plot_image(puzzles[10],"digital_lightsout.png")
    plot_grid(puzzles[:36],"digital_lightsouts.png")
    _transitions = transitions(3)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"digital_lightsout_transitions.png")

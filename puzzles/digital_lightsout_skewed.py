#!/usr/bin/env python3

import numpy as np
from .lightsout import generate_configs, generate_random_configs, successors

on = [[0, 0, 0, 0, 0, ],
      [0, 0, 1, 0, 0, ],
      [0, 1, 1, 1, 0, ],
      [0, 0, 1, 0, 0, ],
      [0, 0, 0, 0, 0, ],]
                            
off= [[0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, ],
      [0, 0, 0, 0, 0, ],]

def generate(configs):
    import math
    size = int(math.sqrt(len(configs[0])))
    base = 5
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
        figure2 = np.zeros((dim*2,dim*2))
        figure2[half:dim+half,half:dim+half] = figure
        figure3 = np.zeros((dim,dim))
        for x in range(-half,half):
            for y in range(-half,half):
                r = math.sqrt(x*x+y*y)
                rad = math.atan2(y,x) + math.pi  * (1-r)/half/6
                p = r * np.array([math.cos(rad), math.sin(rad)])
                px1 = math.floor(p[0])
                py1 = math.floor(p[1])
                grid = np.array([[[px1,py1],[px1+1,py1],],
                                 [[px1,py1+1],[px1+1,py1+1],],])
                w = (1 - np.prod(np.fabs(grid - p),axis=-1))/3
                # print(np.sum(w))
                
                value = np.sum(w * figure2[py1+dim:py1+2+dim,px1+dim:px1+2+dim])
                figure3[y+half,x+half] = value
        return figure3
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim,dim)).clip(0,1)

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
    puzzles = generate([configs[-1]])
    plot_image(puzzles[0],"digital_lightsout_skewed.png")
    # puzzles = generate(configs)
    # plot_image(puzzles[10],"digital_lightsout_skewed.png")
    # plot_grid(puzzles[:36],"digital_lightsout_skeweds.png")
    # _transitions = transitions(3)
    # import numpy.random as random
    # indices = random.randint(0,_transitions[0].shape[0],18)
    # _transitions = _transitions[:,indices]
    # print(_transitions.shape)
    # transitions_for_show = \
    #     np.einsum('ba...->ab...',_transitions) \
    #       .reshape((-1,)+_transitions.shape[2:])
    # print(transitions_for_show.shape)
    # plot_grid(transitions_for_show,"digital_lightsout_skewed_transitions.png")

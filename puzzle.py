#!/usr/bin/env python

import numpy as np


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

def generate_configs(digit=9):
    import itertools
    return itertools.permutations(range(digit))

def generate_puzzle(configs, width, height):
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

def successors(config,width,height):
    pos = config[0]
    x = pos % width
    y = pos // width
    succ = []
    if x is not 0:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos-1)
        c[0] -= 1
        c[other] += 1
        succ.append(c)
    if x is not width-1:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos+1)
        c[0] += 1
        c[other] -= 1
        succ.append(c)
    if y is not 0:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos-width)
        c[0] -= width
        c[other] += width
        succ.append(c)
    if y is not height-1:
        c = list(config)
        other = next(i for i,_pos in enumerate(c) if _pos == pos+width)
        c[0] += width
        c[other] -= width
        succ.append(c)
    return succ

def config_transitions(configs):
    return [ (c1,c2) for c2 in successors(c1) for c1 in configs ]

def states(width, height):
    digit = width * height
    configs = generate_configs(digit)
    return generate_puzzle(configs,width,height)

def transitions(width, height):
    digit = width * height
    configs = generate_configs(digit)
    transitions = np.array([ generate_puzzle([c1,c2],width,height)
                             for c1 in configs for c2 in successors(c1,width,height) ])
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
    configs = generate_configs(6)
    puzzles = generate_puzzle(configs, 2, 3)
    print puzzles[10]
    plot_image(puzzles[10],"samples/puzzle.png")
    plot_grid(puzzles[:36],"samples/puzzles.png")
    _transitions = transitions(2,3)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print _transitions.shape
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print transitions_for_show.shape
    plot_grid(transitions_for_show,"samples/puzzle_transitions.png")

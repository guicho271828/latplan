#!/usr/bin/env python3

import numpy as np
from .puzzle import generate_configs, successors
from .split_image import split_image
import os

from mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]
mnist_panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]

from numpy.random import randint

lenna = None
spider = None

panels = [mnist_panels, lenna, spider]

def random_panels(width,height):
    global lenna, spider, panels
    if lenna is None:
        # 28
        lenna  = split_image(os.path.join(os.path.dirname(__file__), "lenna.png"),width,height)
        stepy = lenna[0].shape[0]//28
        stepx = lenna[0].shape[1]//28
        lenna = lenna[:,::stepy,::stepx][:,:28,:28]
        print(lenna.shape)
        spider  = split_image(os.path.join(os.path.dirname(__file__), "spider.png"),width,height)
        stepy = spider[0].shape[0]//28
        stepx = spider[0].shape[1]//28
        spider = spider[:,::stepy,::stepx][:,:28,:28]
        print(spider.shape)
        panels = [mnist_panels, lenna, spider]
    
    return panels[randint(0,3)]

def generate(configs, width, height, selected_panels):
    assert width*height <= 9
    base_width = selected_panels[0].shape[1]
    base_height = selected_panels[0].shape[0]
    dim_x = base_width*width
    dim_y = base_height*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*base_height:(y+1)*base_height,
                   x*base_width:(x+1)*base_width] = selected_panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

def states(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height,random_panels(width,height))

def transitions(width, height, configs=None, one_per_state=False):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        def pickone(thing):
            index = randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate(
                [c1,pickone(successors(c1,width,height))],width,height,random_panels(width,height))
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],width,height,random_panels(width,height))
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
    puzzles = generate(configs, 2, 3, random_panels(2,3))
    print(puzzles[10])
    plot_image(puzzles[10],"mixed_puzzle.png")
    plot_grid(puzzles[:36],"mixed_puzzles.png")
    _transitions = transitions(2,3)
    import numpy.random as random
    indices = randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"mixed_puzzle_transitions.png")


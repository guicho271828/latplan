#!/usr/bin/env python3

import numpy as np
from .puzzle import generate_configs, successors

from .mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]
panels  = np.array([ imgs[0].reshape((28,28)) for imgs in imgs ])

panels[8] = imgs[8][3].reshape((28,28))
panels[1] = imgs[1][3].reshape((28,28))
panels.append(np.random.uniform(0,1,(28,28)))

base = 14
stepy = panels[0].shape[0]//base
stepx = panels[0].shape[1]//base
panels = panels[:,::stepy,::stepx][:,:base,:base].round()

def generate(configs, width, height):
    assert width*height <= 9
    dim_x = base*width
    dim_y = base*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*base:(y+1)*base,
                   x*base:(x+1)*base] = panels[digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

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
    configs = list(generate_configs(9))[:36]
    puzzles = generate(configs, 3, 3)
    plot_grid(puzzles,"mnist_puzzles.png")
    _transitions = transitions(3,3,configs)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"mnist_puzzle_transitions.png")


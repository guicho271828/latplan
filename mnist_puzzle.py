#!/usr/bin/env python

import numpy as np
from puzzle import generate_configs

from mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]
panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]

def generate_mnist_puzzle(configs, width, height):
    assert width*height <= 9
    base_width = 28
    base_height = 28
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

def states(width, height):
    digit = width * height
    configs = generate_configs(digit)
    return generate_mnist_puzzle(configs,width,height)

def transitions(width, height):
    from puzzle import successors
    digit = width * height
    configs = generate_configs(digit)
    transitions = np.array([ generate_mnist_puzzle([c1,c2],width,height)
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
    puzzles = generate_mnist_puzzle(configs, 2, 3)
    print puzzles[10]
    plot_image(puzzles[10],"samples/mnist_puzzle.png")
    plot_grid(puzzles[:36],"samples/mnist_puzzles.png")
    _transitions = transitions(2,3)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print _transitions.shape
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print transitions_for_show.shape
    plot_grid(transitions_for_show,"samples/mnist_puzzle_transitions.png")


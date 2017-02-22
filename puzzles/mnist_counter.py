#!/usr/bin/env python3

import numpy as np
from .counter import generate_configs, successors

from .mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]
panels  = [ imgs[0].reshape((28,28)) for imgs in imgs ]

def generate(configs):
    return np.array([ panels[c] for c in configs ])

def states(size,configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    transitions = np.array([ generate([c1,c2])
                             for c1 in configs for c2 in successors(c1,size) ])
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
        h = 1+ (l//6)
        plt.figure(figsize=(20, h*2))
        for i,image in enumerate(images):
            # display original
            ax = plt.subplot(h,w,i+1)
            plt.imshow(image,interpolation='nearest',cmap='gray',)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(name)
    configs = generate_configs(10)
    puzzles = generate(configs)
    print(puzzles[9])
    plot_image(puzzles[9],"mnist_counter.png")
    plot_grid(puzzles[:36],"mnist_counters.png")
    _transitions = transitions(10)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"mnist_counter_transitions.png")

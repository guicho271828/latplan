#!/usr/bin/env python3
import numpy as np

from mnist import mnist
x_train, y_train, _, _ = mnist()
filters = [ np.equal(i,y_train) for i in range(10) ]
imgs    = [ x_train[f] for f in filters ]

examples = 10
panels = []
for i in range(examples):
    panels.extend([ imgs[i].reshape((28,28)) for imgs in imgs ])

import matplotlib.pyplot as plt

def plot_grid(images,name="plan.png"):
    import matplotlib.pyplot as plt
    l = len(images)
    w = 10
    h = examples
    plt.figure(figsize=(20, h*2))
    for i,image in enumerate(images):
        # display original
        ax = plt.subplot(h,w,i+1)
        plt.imshow(image,interpolation='nearest',cmap='gray',)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(name)

plot_grid(panels,"test-mnist.png")

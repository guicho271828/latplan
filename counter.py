#!/usr/bin/env python3

# from __future__ import absolute_import
from mnist import mnist
import numpy as np

def states(labels = range(10)):
    x_train, _, _,_ = mnist(labels)
    return x_train.reshape((-1,28,28))

def transitions(labels = range(10), n = 10000):
    "return pairs of indices of images that contain adjacent digits"

    pairs = []
    first = True
    prev = None
    for label in labels:
        if first:
            first = False
            prev = label
        else:
            pairs.append((prev,label))
            pairs.append((label,prev))
    
    x_train, y_train, _,_ = mnist(labels)
    labelled_images = {}
    for label in labels:
        filter = np.equal(y_train,label)
        labelled_images[label] = x_train[filter]

    orig_results = []
    dest_results = []
    for orig,dest in pairs:
        orig_results.append(
            labelled_images[orig][np.random.choice(len(labelled_images[orig]),n)])
        dest_results.append(
            labelled_images[dest][np.random.choice(len(labelled_images[dest]),n)])
    return np.array((np.concatenate(orig_results,axis=0),
                     np.concatenate(dest_results,axis=0))).reshape((2,-1,28,28))

if __name__ == '__main__':
    print(transitions(n=10).shape)

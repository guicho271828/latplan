#!/usr/bin/env python

# from __future__ import absolute_import
from mnist import mnist
import numpy as np

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
    
    return np.concatenate(orig_results,axis=0), np.concatenate(dest_results,axis=0),

if __name__ == '__main__':
    print transitions(n=10)

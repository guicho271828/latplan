#!/usr/bin/env python3

import numpy as np

def generate_configs (size=3):
    import itertools
    return list(itertools.product([-1,1], repeat=size*size))

def generate_random_configs (size=3,sample=10000):
    return np.random.choice([-1,1],(sample,size*size))

def successors (config):
    import math
    leds = len(config)
    size = int(math.sqrt(leds))
    succs = []
    for i in range(leds):
        y = i // size
        x = i % size
        succ = np.copy(config)
        succ[i] *= -1
        if x-1 >= 0:
            succ[i-1] *= -1
        if x+1 < size:
            succ[i+1] *= -1
        if y-1 >= 0:
            succ[i-size] *= -1
        if y+1 < size:
            succ[i+size] *= -1
        succs.append(succ)
    return succs


if __name__ == '__main__':
    print(np.array(list(successors(np.ones(9,dtype=np.int8)))).reshape((-1,3,3)))
    

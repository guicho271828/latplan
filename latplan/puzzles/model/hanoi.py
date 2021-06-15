#!/usr/bin/env python3

import numpy as np

# config encoding:
# A config is a sequence of x-positions (towers). An example with 6 disks, 3 towers is
# [0,2,1,2,1,0]
# since larger disks cannot be above smaller disks,
# the assignments of each disk to some tower defines a unique, valid state, e.g.,
# [0,2,1,2,1,0] == [[05][24][13]]

# generate all configs
def generate_configs(disks=6,towers=3):
    import itertools
    return itertools.product(range(towers),repeat=disks)

# sample configs
def generate_random_configs(disks=6,towers=3,sample=10000):
    return np.random.randint(0,towers,(sample,disks))

# state encoding:
# intermediate representation for generating an image.
# (it is just easier to handle from the renderer's point of view)

# each tower has a sequence of numbers (disks)
# in the decreasing order
# for example,
# [[012345678][][]] is the initial state of the tower
# [[][][012345678]] is the goal state of the tower

def config_state(config,disks,towers):
    disk_state = []
    for _ in range(towers):
        disk_state.append([])
    for disk,pos in enumerate(config):
        disk_state[pos].append(disk)
    return disk_state

def state_config(state,disks,towers):
    config = np.zeros(disks,dtype=np.int8)
    for i,tower in enumerate(state):
        for disk in tower:
            config[disk] = i
    return config

def successors(config,disks,towers):
    from copy import deepcopy
    state = config_state(config,disks,towers)
    succ = []
    for i in range(towers):
        for j in range(towers):
            if j != i \
               and len(state[i]) > 0 \
               and ( len(state[j]) == 0 or state[j][0] > state[i][0] ):
                # pseudo code
                copy = deepcopy(state)
                disk = copy[i].pop(0)
                copy[j].insert(0,disk)
                succ.append(state_config(copy,disks,towers))
    return succ


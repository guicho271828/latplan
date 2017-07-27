#!/usr/bin/env python3

import numpy as np

# config encoding:
# A config is a sequence of x-positions (towers)
# [0,2,1,2,1,0]
# since there is an implicit order that is enforced,
# the assignments of each disk to some tower defines a unique, valid state
# [0,2,1,2,1,0] == [[05][24][13]]

# random config is available from np.random.randint(0,3,size)

def generate_configs(disks=6,towers=3):
    import itertools
    return itertools.product(range(towers),repeat=disks)

# state encoding:
# intermediate representation for generating an image.
# XX each disk has an x-position and a y-position (counted from the bottom)
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


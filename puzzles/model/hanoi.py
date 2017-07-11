#!/usr/bin/env python3

import numpy as np

# state encoding:
# XX each disk has an x-position and a y-position (counted from the bottom)
# 3 towers
# each tower has a sequence of numbers (disks)
# in the decreasing order
# for example,
# [[012345678][][]] is the initial state of the tower
# [[][][012345678]] is the goal state of the tower

# how to enumerate such thing:

# each disk has a position and it also defines the state
# but many/most positions are invalid?
# [0,2,1,2,1,0]
# no, since there is an implicit order that is enforced,
# the assignments of each disk to some tower defines the entire state
# [0,2,1,2,1,0] == [[05][24][13]]

# random config is available from np.random.randint(0,3,size)

def generate_configs(size=6):
    import itertools
    return itertools.product(range(3),repeat=size)

def config_state(config):
    disk_state =[[],[],[]]
    for disk,pos in enumerate(config):
        disk_state[pos].append(disk)
    return disk_state

def state_config(state):
    size = len(state[0]) + len(state[1]) + len(state[2])
    config = np.zeros(size,dtype=np.int8)
    for i in range(3):
        for disk in state[i]:
            config[disk] = i
    return config

def successors(config):
    from copy import deepcopy
    # at most 6 successors
    state = config_state(config)
    succ = []
    for i in range(3):
        for j in range(3):
            if j != i and state[i]:
                if not state[j] or state[j][0] > state[i][0]:
                    # pseudo code
                    copy = deepcopy(state)
                    disk = copy[i].pop(0)
                    copy[j].append(disk)
                    succ.append(state_config(copy))
    return succ


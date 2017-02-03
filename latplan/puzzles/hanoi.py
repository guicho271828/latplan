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

def generate_configs(size=10):
    assert size < 21
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
                    disk = copy[i].pop()
                    copy[j].append(disk)
                    succ.append(state_config(copy))
    return succ

def generate1(config):
    figure = np.zeros([84,84],dtype=np.int8)
    state = config_state(config)
    for i, tower in enumerate(state):
        y = 1
        tower.reverse()
        for disk in tower:
            figure[81-y:83-y,(28*i-14)-disk:(28*i-14)+disk] = 1
            y += 4
    return figure

def generate(configs):
    return np.array([ generate1(c) for c in configs ])
                

def states(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None, one_per_state=False):
    if configs is None:
        configs = generate_configs(size)
    transitions = np.array([
        generate([c1,c2])
        for c1 in configs for c2 in successors(c1)])
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
    disks = 9
    configs = generate_configs(disks)
    puzzles = generate(configs)
    print(puzzles.shape)
    print(puzzles[10])
    plot_image(puzzles[10],"hanoi.png")
    plot_grid(puzzles[:36],"hanois.png")
    _transitions = transitions(disks)
    print(_transitions.shape)
    import numpy.random as random
    indices = random.randint(0,_transitions[0].shape[0],18)
    _transitions = _transitions[:,indices]
    print(_transitions.shape)
    transitions_for_show = \
        np.einsum('ba...->ab...',_transitions) \
          .reshape((-1,)+_transitions.shape[2:])
    print(transitions_for_show.shape)
    plot_grid(transitions_for_show,"hanoi_transitions.png")


#!/usr/bin/env python3
import warnings
import config
import numpy as np
from model import Discriminator, default_networks

import keras.backend as K
import tensorflow as tf
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

sd = None
ad = None
ae = None
max_diff = None

OPEN   = 0
CLOSED = 1

# class State(object):
#     def __init__(self, raw_state):
#         self.raw_state  = raw_state
# 
#     def __hash__(self):
#         return np.sum(self.raw_state << np.arange(len(self.raw_state)))
# 
#     def __repr__(self):
#         return "State({})".format(self.raw_state)

def state_hash(state):
    return np.sum(state << np.arange(len(state)))

class State(object):
    def __init__(self, state, g=math.inf, parent=None, status=OPEN):
        self.state  = state
        self.g      = g
        self.parent = parent
        self.status = status

    def __hash__(self):
        return state_hash(self.state)

    def __repr__(self):
        return "State({},{},{},{})".format(self.state,self.g,self.parent,self.status)

    def path(self):
        if self.parent:
            return [self.state, *self.parent.path()]
        else:
            return [self.state]
    
def astar(init,goal,h):
    
    import queue
    open_list = queue.PriorityQueue()

    open_list.put((0, h(init,goal), State(init, 0)))

    close_list = {}

    import itertools
    N = len(init)
    print("building a flipper")
    flip_indices = np.array(list(itertools.combinations(list(range(N)), max_diff)))
    print("building a flipper")
    flipper = np.zeros((len(flip_indices),N),dtype=np.uint8)
    print("building a flipper")
    for i in range(len(flip_indices)):
        flipper[i,flip_indices[i]] = 1
    print("flipper building finished; elements:", len(flipper))
    
    def successors(state):
        s = state.state
        all_possible_succ = s ^ flipper
        valid_possible_succ = all_possible_succ[np.where(0.8 < sd.discriminate(all_possible_succ,batch_size=4000))]
        print("valid_possible_succ",len(valid_possible_succ))
        actions = np.concatenate((s,valid_possible_succ),axis=1)
        valid_actions = actions[np.where(0.8 < ad.discriminate(actions,batch_size=4000))]
        print("valid_actions",len(valid_actions))
        valid_succ = valid_actions[:,N:]
        for succ in valid_succ:
            hash_value = state_hash(succ)
            if hash_value in close_list:
                yield close_list[hash_value]
            else:
                yield State(succ)

    best_f = -1
    best_h = math.inf
    while True:
        f, h, state = open_list.get()

        if state.status == CLOSED:
            continue
        
        state.status = CLOSED

        if best_f < f:
            best_f = f
            print("new f = {}".format(f))
        if best_h > h:
            best_h = h
            print("new h = {}".format(h))

        if (state.state == goal).all():
            return state

        for c in successors(state):
            new_g = state.g + 1
            if c.g > new_g:
                c.g      = new_g
                c.parent = state
                c.status = OPEN
                open_list.put((0, h(c,goal), c))
            
        
def goalcount(state,goal):
    return np.abs(state-goal).sum()

def main(directory, init_path, goal_path):
    global sd, ad, ae, max_diff
    
    sd = Discriminator("{}/_sd/".format(directory)).load()
    ad = Discriminator("{}/_ad/".format(directory)).load()

    from latplan.util import get_ae_type
    ae = default_networks[get_ae_type(directory)](directory).load()

    known_actions = np.loadtxt(ae.local("actions.csv"),dtype=np.int8)
    N = known_actions.shape[1]//2
    pre, suc = known_actions[:,:N], known_actions[:,N:]
    abs_diff = np.sum(np.abs(pre-suc),axis=1)
    print(np.histogram(abs_diff,N,(0,N))[0])
    max_diff = np.max(abs_diff)
    
    from scipy import misc
    init_image = misc.imread(init_path)
    goal_image = misc.imread(goal_path)
    
    init = ae.encode_binary(np.expand_dims(init_image,0))[0]
    goal = ae.encode_binary(np.expand_dims(goal_image,0))[0]

    print(astar(init,goal,goalcount).path())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        sys.exit("{} [networkdir] [init.png] [goal.png]".format(sys.argv[0]))
    main(*sys.argv[1:])



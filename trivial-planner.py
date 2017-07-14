#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import default_networks, ActionAE

import keras.backend as K
import tensorflow as tf
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

sae = None
oae = None
available_actions = None

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

    N = len(init)
    
    def successors(state):
        s = state.state
        y = oae.decode([np.repeat(s, len(available_actions)), available_actions])
        t = y[:,N:]
        # for now, assume they are all valid
        for succ in t:
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
    global sae, oae, available_actions
    
    from latplan.util import get_ae_type
    sae = default_networks[get_ae_type(directory)](directory).load()
    oae = ActionAE(sae.local("_aae/")).load()
    
    known_transisitons = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    print(known_transisitons,known_transisitons.shape)
    actions = oae.encode_action(known_transisitons, batch_size=1000)
    print(actions) 
    print(actions[0], actions[0].sum(axis=1))
    histogram = actions.sum(axis=0,dtype=int)
    print(histogram)
    print(np.count_nonzero(histogram))
    available_actions = np.zeros((np.count_nonzero(histogram), actions.shape[1]))
    available_actions[np.where(histogram > 0)] = 1
    print(available_actions)
    
    from scipy import misc
    init_image = misc.imread(init_path)
    goal_image = misc.imread(goal_path)
    
    init = sae.encode_binary(np.expand_dims(init_image,0))[0]
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0]
    
    print(astar(init,goal,goalcount).path())

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        sys.exit("{} [networkdir] [init.png] [goal.png]".format(sys.argv[0]))
    main(*sys.argv[1:])



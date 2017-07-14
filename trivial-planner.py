#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import default_networks, ActionAE, Discriminator

import keras.backend as K
import tensorflow as tf
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

sae = None
oae = None
ad  = None
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
            return [*self.parent.path(),self.state]
        else:
            return [self.state]
    
def astar(init,goal,heuristic):
    
    N = len(init)
    
    import queue
    open_list = queue.PriorityQueue()
    open_list.put((0, heuristic(init,goal), state_hash(init)))

    close_list = {}
    close_list[state_hash(init)] = State(init, 0)
    
    def successors(state):
        s = state.state
        y = oae.decode([np.repeat(np.expand_dims(s,0), len(available_actions), axis=0), available_actions]) \
               .round().astype(int)
        valid_y = y[np.where(np.squeeze(ad.discriminate(y)) > 0.8)[0]]
        t = valid_y[:,N:]

        # filtering based on reconstruction
        images  = sae.decode_binary(t)
        images2 = sae.autoencode(images)
        binary_crossentropy = - (images     * np.log(images2+1e-5) + \
                                 (1-images) * np.log(1-images2+1e-5)).mean(axis=(1,2))
        valid_t = t[np.where(binary_crossentropy < 0.01)]
        
        print(len(y),"->",len(valid_y),"->",len(valid_t))
        # for now, assume they are all valid
        for i,succ in enumerate(valid_t):
            # print(succ)
            hash_value = state_hash(succ)
            if hash_value in close_list:
                yield close_list[hash_value]
            else:
                _succ = State(succ)
                close_list[hash_value] = _succ
                yield _succ

    best_f = -1
    best_h = math.inf
    while True:
        f, h, shash = open_list.get()
        state = close_list[shash]
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
                open_list.put((0, heuristic(c.state,goal), state_hash(c.state)))
            
        
def goalcount(state,goal):
    return np.abs(state-goal).sum()

def main(directory, init_path, goal_path):
    global sae, oae, ad, available_actions
    
    from latplan.util import get_ae_type
    sae = default_networks[get_ae_type(directory)](directory).load()
    oae = ActionAE(sae.local("_aae/")).load()
    ad  = Discriminator(sae.local("_ad/")).load()
    
    known_transisitons = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    actions = oae.encode_action(known_transisitons, batch_size=1000).round()
    histogram = np.squeeze(actions.sum(axis=0,dtype=int))
    print(histogram)
    print(np.count_nonzero(histogram),"actions valid")
    print("valid actions:")
    print(np.where(histogram > 0)[0])
    identified, total = np.squeeze(histogram.sum()), len(actions)
    if total != identified:
        print("network does not explain all actions: only {} out of {} ({}%)".format(
            identified, total, identified * 100 // total ))
    available_actions = np.zeros((np.count_nonzero(histogram), actions.shape[1], actions.shape[2]), dtype=int)
    for i, pos in enumerate(np.where(histogram > 0)[0]):
        available_actions[i][0][pos] = 1
    
    from scipy import misc
    init_image = misc.imread(init_path)
    goal_image = misc.imread(goal_path)
    
    init = sae.encode_binary(np.expand_dims(init_image,0))[0]
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0]

    path = np.array(astar(init,goal,goalcount).path())
    print(path)
    from latplan.util.plot import plot_grid
    plot_grid(sae.decode_binary(path),path="path.png",verbose=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 4:
        sys.exit("{} [networkdir] [init.png] [goal.png]".format(sys.argv[0]))
    main(*sys.argv[1:])


def test():
    main("samples/puzzle_mnist33_fc/",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/init.png",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/goal.png")
    

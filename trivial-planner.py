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
sd  = None
ad2  = None
sd2  = None
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

def bce(x,y,axis):
    return - (x * np.log(y+1e-5) + \
              (1-x) * np.log(1-y+1e-5)).mean(axis=axis)

def absolute_error(x,y,axis):
    return np.sum(np.absolute(x - y),axis=axis)


def state_hash(state):
    return np.sum(state << np.arange(len(state)))

class State(object):
    def __init__(self, state, g=math.inf, h=0, parent=None, status=OPEN):
        self.state  = state
        self.g      = g
        self.h      = h
        self.parent = parent
        self.status = status

    def hash(self):
        return state_hash(self.state)

    def __repr__(self):
        return "State({},{},{},{})".format(self.state,self.g,self.parent,self.status)

    def path(self):
        if self.parent:
            return [*self.parent.path(),self.state]
        else:
            return [self.state]
    
def astar(init,goal,distance):
    
    N = len(init)
    heuristic = lambda x: distance(x, goal)
    
    _init = State(init, g=0, h=heuristic(init))
    
    import queue
    open_list = queue.PriorityQueue()
    open_list.put((_init.g + _init.h, _init.h, _init.hash()))

    close_list = {}
    close_list[_init.hash()] = _init
    
    def successors(state):
        reductions = []
        s = state.state
        y = oae.decode([np.repeat(np.expand_dims(s,0), len(available_actions), axis=0), available_actions]) \
               .round().astype(int)
        reductions.append(len(y))
        
        # filtering based on OAE action reconstruction
        action_reconstruction = oae.encode_action(y).round()
        # loss = bce(available_actions, action_reconstruction, (1,2))
        loss = absolute_error(available_actions, action_reconstruction, (1,2))
        # print(loss)
        y = y[np.where(loss < 0.01)]
        reductions.append(len(y))
        
        # filtering based on Action Discriminator
        y = y[np.where(np.squeeze(ad.discriminate(y)) > 0.8)[0]]
        reductions.append(len(y))

        t = y[:,N:]

        if len(y) == 0:
            return
        # filtering based on SAE reconstruction
        images  = sae.decode_binary(t).round()
        images2 = sae.autoencode(images).round()
        loss = absolute_error(images,images2,(1,2))
        # print(loss)
        t = t[np.where(loss < 0.01)].astype(int)
        reductions.append(len(t))

        # filtering based on State Discriminator
        t = t[np.where(np.squeeze(sd.discriminate(t)) > 0.8)[0]]
        reductions.append(len(t))
        
        print("->".join(map(str,reductions)))
        # for now, assume they are all valid
        for i,succ in enumerate(t):
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
                c.h      = heuristic(c.state)
                open_list.put((c.g+c.h, c.h, c.hash()))
            
        
def goalcount(state,goal):
    return np.abs(state-goal).sum()

def main(network_dir, problem_dir):
    global sae, oae, ad, ad2, sd, sd2, available_actions
    
    from latplan.util import get_ae_type
    sae = default_networks[get_ae_type(network_dir)](network_dir).load()
    oae = ActionAE(sae.local("_aae/")).load()
    ad  = Discriminator(sae.local("_ad/")).load()
    sd  = Discriminator(sae.local("_sd/")).load()
    ad2  = Discriminator(sae.local("_ad2/")).load(allow_failure=True)
    sd2  = Discriminator(sae.local("_sd2/")).load(allow_failure=True)
    
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
    import os.path
    # d, n = os.path.split(problem_dir)
    # if n == '':
    #     problem_dir = d
    # else:
    #     problem_dir = 
    
    init_image = misc.imread(os.path.join(problem_dir,"init.png"))
    goal_image = misc.imread(os.path.join(problem_dir,"goal.png"))
    
    init = sae.encode_binary(np.expand_dims(init_image,0))[0].astype(int)
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0].astype(int)

    plan = np.array(astar(init,goal,goalcount).path())
    print(plan)
    from latplan.util.plot import plot_grid
    plot_grid(sae.decode_binary(plan),path=os.path.join(problem_dir,"path.png"),verbose=True)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        sys.exit("{} [networkdir] [problemdir]".format(sys.argv[0]))
    main(*sys.argv[1:])


def test():
    # ./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/
    main("samples/puzzle_mnist33_fc/",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/")
    

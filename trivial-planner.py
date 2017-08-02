#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import default_networks, ActionAE, Discriminator, PUDiscriminator, combined_discriminate, combined_discriminate2
from latplan.util import get_ae_type
from latplan.util.plot import plot_grid
import os.path
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
sd3 = None
cae = None

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

def action_reconstruction_filtering(y):
    # filtering based on OAE action reconstruction
    action_reconstruction = oae.encode_action(y).round()
    # loss = bce(available_actions, action_reconstruction, (1,2))
    loss = absolute_error(available_actions, action_reconstruction, (1,2))
    # print(loss)
    return y[np.where(loss < 0.01)]

def state_reconstruction_from_oae_filtering(y):
    action_reconstruction = oae.encode_action(y)
    N = y.shape[1]//2
    y_reconstruction = oae.decode([y[:,:N], action_reconstruction])
    loss = absolute_error(y, y_reconstruction, (1,))
    # print(loss)
    return y[np.where(loss < 0.01)]

def action_discriminator_filtering(y):
    return y[np.where(np.squeeze(ad.discriminate(y)) > 0.5)[0]]

action_pruning_methods = [action_reconstruction_filtering,
                          # state_reconstruction_from_oae_filtering,
                          action_discriminator_filtering]

def state_reconstruction_filtering(t):
    # filtering based on SAE reconstruction
    images  = sae.decode_binary(t).round()
    images2 = sae.autoencode(images).round()
    loss = absolute_error(images,images2,(1,2))
    # print(loss)
    return t[np.where(loss < 0.01)].astype(int)
    
def state_discriminator_filtering(t):
    # filtering based on State Discriminator
    return t[np.where(np.squeeze(sd.discriminate(t)) > 0.5)[0]]

def state_discriminator3_filtering(t):
    if "conv" in get_ae_type(sae.path):
        return t[np.where(np.squeeze(combined_discriminate2(t,sae,sd3)) > 0.5)[0]]
    else:
        return t[np.where(np.squeeze(combined_discriminate(t,sae,cae,sd3)) > 0.5)[0]]

state_pruning_methods = [state_reconstruction_filtering,
                         state_discriminator3_filtering]

class Searcher:
    def successors(self,state):
        reductions = []
        s = state.state
        y = oae.decode([np.repeat(np.expand_dims(s,0), len(available_actions), axis=0), available_actions]) \
               .round().astype(int)
        reductions.append(len(y))
        for m in action_pruning_methods:
            y = m(y)
            reductions.append(len(y))
            if len(y) == 0:
                return

        t = y[:,self.N:]

        for m in state_pruning_methods:
            t = m(t)
            reductions.append(len(t)) 
            if len(t) == 0:
                return

        print("->".join(map(str,reductions)))
        # for now, assume they are all valid
        for i,succ in enumerate(t):
            # print(succ)
            hash_value = state_hash(succ)
            if hash_value in self.close_list:
                yield self.close_list[hash_value]
            else:
                _succ = State(succ)
                self.close_list[hash_value] = _succ
                yield _succ

class Astar(Searcher):
    def search(self,init,goal,distance):
        self.N = len(init)
        heuristic = lambda x: distance(x, goal)

        _init = State(init, g=0, h=heuristic(init))

        import queue
        open_list = queue.PriorityQueue()
        open_list.put((_init.g + _init.h, _init.h, _init.hash()))

        self.close_list = {}
        self.close_list[_init.hash()] = _init

        best_f = -1
        best_h = math.inf
        while True:
            if open_list.empty():
                raise Exception("Open list is empty!")
            f, h, shash = open_list.get()
            state = self.close_list[shash]
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

            for c in self.successors(state):
                new_g = state.g + 1
                if c.g > new_g:
                    c.g      = new_g
                    c.parent = state
                    c.status = OPEN
                    c.h      = heuristic(c.state)
                    open_list.put((c.g+c.h, c.h, c.hash()))

class GBFS(Searcher):
    def search(self,init,goal,distance):
        self.N = len(init)
        heuristic = lambda x: distance(x, goal)

        _init = State(init, g=0, h=heuristic(init))

        import queue
        open_list = queue.PriorityQueue()
        open_list.put((_init.h, _init.hash()))

        self.close_list = {}
        self.close_list[_init.hash()] = _init

        best_h = math.inf
        while True:
            if open_list.empty():
                raise Exception("Open list is empty!")
            h, shash = open_list.get()
            state = self.close_list[shash]
            if state.status == CLOSED:
                continue

            state.status = CLOSED

            if best_h > h:
                best_h = h
                print("new h = {}".format(h))

            if (state.state == goal).all():
                return state

            for c in self.successors(state):
                new_g = state.g + 1
                if c.g > new_g:
                    c.g      = new_g
                    c.parent = state
                    c.status = OPEN
                    c.h      = heuristic(c.state)
                    open_list.put((c.h, c.hash()))
        
def goalcount(state,goal):
    return np.abs(state-goal).sum()

def main(network_dir, problem_dir):
    global sae, oae, ad, ad2, sd, sd2, sd3, cae, available_actions
    
    sae = default_networks[get_ae_type(network_dir)](network_dir).load()
    oae = ActionAE(sae.local("_aae/")).load()
    ad  = PUDiscriminator(sae.local("_ad/")).load()
    # sd  = Discriminator(sae.local("_sd/")).load(allow_failure=True)
    # ad2 = Discriminator(sae.local("_ad2/")).load(allow_failure=True)
    # sd2 = Discriminator(sae.local("_sd2/")).load(allow_failure=True)
    cae = default_networks['SimpleCAE'](sae.local("_cae/")).load(allow_failure=True)
    sd3 = PUDiscriminator(sae.local("_sd3/")).load()

    def problem(path):
        return os.path.join(problem_dir,path)
    def network(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(root, get_ae_type(network_dir), ext)
    
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

    init_image = misc.imread(problem("init.png"))
    goal_image = misc.imread(problem("goal.png"))
    
    init = sae.encode_binary(np.expand_dims(init_image,0))[0].round().astype(int)
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0].round().astype(int)
    print(init)
    print(goal)
    plot_grid(
        sae.decode_binary(np.array([init,goal])),
        path=problem(network("init_goal_reconstruction.png")),verbose=True)
    plan = np.array(GBFS().search(init,goal,goalcount).path())
    print(plan)
    plot_grid(sae.decode_binary(plan),
              path=problem(network("path.png")),verbose=True)

    from latplan.util import ensure_directory
    module_name = ensure_directory(problem_dir).split("/")[-3]
    from importlib import import_module
    m = import_module(module_name)
    m.setup()

    validation = m.validate_transitions([sae.decode_binary(plan[0:-1]), sae.decode_binary(plan[1:])],3,3)
    print(validation)
    print(ad.discriminate( np.concatenate((plan[0:-1], plan[1:]), axis=-1)).flatten())
    import subprocess
    subprocess.call(["rm", "-f", problem(network("path.valid"))])
    import sys
    if np.all(validation):
        subprocess.call(["touch", problem(network("path.valid"))])
        sys.exit(0)
    else:
        sys.exit(2)
        

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        sys.exit("{} [networkdir] [problemdir]".format(sys.argv[0]))
    main(*sys.argv[1:])


def test():
    # ./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/
    main("samples/puzzle_mnist33_fc/",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/")
    

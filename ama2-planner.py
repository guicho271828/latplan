#!/usr/bin/env python3
import warnings
import sys
import numpy as np
import latplan
import latplan.model
from latplan.model import combined_sd
from latplan.util import *
from latplan.util.plot import *
from latplan.util.np_distances import *
from latplan.util.planner import *
import os.path
import keras.backend as K
import tensorflow as tf
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

sae = None
aae = None
ad  = None
sd3 = None
cae = None

available_actions = None
inflation = 5

image_threshold = 0.1
image_diff = mae

OPEN   = 0
CLOSED = 1

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
    # filtering based on AAE action reconstruction
    action_reconstruction = aae.encode_action(y).round()
    return y[np.all(np.equal(available_actions, action_reconstruction),axis=(1,2))]

def state_reconstruction_from_aae_filtering(y):
    action_reconstruction = aae.encode_action(y)
    N = y.shape[1]//2
    y_reconstruction = aae.decode([y[:,:N], action_reconstruction]).round()
    return y[np.all(np.equal(y, y_reconstruction),axis=(1,))]

def state_reconstruction_filtering(y):
    # filtering based on SAE state reconstruction
    N = y.shape[1]//2
    t = y[:,N:]
    t_reconstruction = sae.autodecode(t).round()
    return y[np.all(np.equal(t, t_reconstruction),axis=(1,))]

def inflate_actions(y):
    from latplan.util import union
    N = y.shape[1]//2
    t = y[:,N:]
    for i in range(inflation-1):
        t = union(sae.autodecode(t).round().astype(int), t)
    y = y.repeat(math.ceil(len(t)/len(y)), axis=0)[:len(t)]
    y[:,N:] = t
    return y

def action_discriminator_filtering(y):
    return y[np.where(np.squeeze(ad.discriminate(y)) > 0.5)[0]]
    
def state_discriminator_filtering(y):
    # filtering based on State Discriminator
    N = y.shape[1]//2
    return y[np.where(np.squeeze(sd.discriminate(y[:,N:])) > 0.5)[0]]

def state_discriminator3_filtering(y):
    N = y.shape[1]//2
    return y[np.where(np.squeeze(combined_sd(y[:,N:],sae,cae,sd3)) > 0.5)[0]]

def cheating_validation_filtering(y):
    N = y.shape[1]//2
    p = puzzle_module(sae)
    pre_images = sae.decode(y[:,:N],batch_size=1000)
    suc_images = sae.decode(y[:,N:],batch_size=1000)
    return y[p.validate_transitions([pre_images, suc_images],batch_size=1000)]

def cheating_image_reconstruction_filtering(y):
    N = y.shape[1]//2
    # filtering based on SAE reconstruction
    images  = sae.decode(y[:,N:]).round()
    images2 = sae.autoencode(images).round()
    loss = image_diff(images,images2,(1,2))
    # print(loss)
    return y[np.where(loss < image_threshold)].astype(int)

pruning_methods = None

def decide_pruning_method():
    # Ad-hoc improvement: if the state discriminator type-1 error is very high
    # (which is not cheating because it can be verified from the training
    # dataset), don't include SD pruning. The threshold misclassification rate
    # is arbitrarily set as 0.25 .

    global pruning_methods
    print("verifying SD type-1 error")
    states_valid = np.loadtxt(sae.local("states.csv"),dtype=np.int8)
    type1_d = combined_sd(states_valid,sae,cae,sd3)
    type1_error = np.sum(1- type1_d) / len(states_valid)
    if type1_error > 0.25:
        pruning_methods = [
            action_reconstruction_filtering,           # if applied, this should be the first method
            # state_reconstruction_from_aae_filtering,
            # inflate_actions,
            action_discriminator_filtering,
            state_reconstruction_filtering,
        ]
    else:
        pruning_methods = [
            action_reconstruction_filtering,           # if applied, this should be the first method
            # state_reconstruction_from_aae_filtering,
            # inflate_actions,
            action_discriminator_filtering,
            state_reconstruction_filtering,
            state_discriminator3_filtering,
        ]

class Searcher:
    def __init__(self):
        import queue
        self.open_list = queue.PriorityQueue()
        self.close_list = {}
        self.stats = {
            "statistics":{
                "generated_including_duplicate":1, # init
                "generated":1,
                "expanded":0,
                "reopened":0,
            }
        }

    def successors(self,state):
        self.stats["statistics"]["expanded"] += 1
        try:
            reductions = []
            s = state.state
            y = aae.decode([np.repeat(np.expand_dims(s,0), len(available_actions), axis=0), available_actions]) \
                   .round().astype(int)
            reductions.append(len(y))

            for m in pruning_methods:
                y = m(y)
                reductions.append(len(y))
                if len(y) == 0:
                    return

            # for now, assume they are all valid
            for i,succ in enumerate(y[:,self.N:]):
                # print(succ)
                self.stats["statistics"]["generated_including_duplicate"] += 1
                hash_value = state_hash(succ)
                if hash_value in self.close_list:
                    yield self.close_list[hash_value]
                else:
                    self.stats["statistics"]["generated"] += 1
                    _succ = State(succ)
                    self.close_list[hash_value] = _succ
                    yield _succ
        finally:
            print("->".join(map(str,reductions)))

    def report(self,path):
        for k, v in self.stats.items():
            print(k, v)
        import json
        with open(path,"w") as f:
            json.dump(self.stats, f)

class StateBasedGoalDetection:
    def goalp(self,state,goal):
        return (state == goal).all()

class ReconstructionGoalDetection:
    def goalp(self,state,goal):
        return image_diff(sae.decode_binary(np.expand_dims(state,0)),
                          sae.decode_binary(np.expand_dims(goal, 0))) < image_threshold

class Astar(Searcher,StateBasedGoalDetection):
    def search(self,init,goal,distance):
        self.N = len(init)
        heuristic = lambda x: distance(x, goal)

        _init = State(init, g=0, h=heuristic(init))

        self.open_list.put((_init.g + _init.h, _init.h, _init.hash()))
        self.close_list[_init.hash()] = _init

        best_f = -1
        best_h = math.inf
        while True:
            if self.open_list.empty():
                raise Exception("Open list is empty!")
            f, h, shash = self.open_list.get()
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

            if self.goalp(state.state, goal):
                yield state
            else:
                for c in self.successors(state):
                    new_g = state.g + 1
                    if c.g > new_g:
                        if c.status == CLOSED:
                            self.stats["statistics"]["reopened"] += 1
                        c.g      = new_g
                        c.parent = state
                        c.status = OPEN
                        c.h      = heuristic(c.state)
                        self.open_list.put((c.g+c.h, c.h, c.hash()))

class GBFS(Searcher,StateBasedGoalDetection):
    def search(self,init,goal,distance):
        self.N = len(init)
        heuristic = lambda x: distance(x, goal)

        _init = State(init, g=0, h=heuristic(init))

        self.open_list.put((_init.h, _init.hash()))
        self.close_list[_init.hash()] = _init

        best_h = math.inf
        while True:
            if self.open_list.empty():
                raise Exception("Open list is empty!")
            h, shash = self.open_list.get()
            state = self.close_list[shash]
            if state.status == CLOSED:
                continue

            state.status = CLOSED

            if best_h > h:
                best_h = h
                print("new h = {}".format(h))

            if self.goalp(state.state, goal):
                yield state
            else:
                for c in self.successors(state):
                    new_g = state.g + 1
                    if c.g > new_g and c.status == OPEN:
                        c.g      = new_g
                        c.parent = state
                        c.status = OPEN
                        c.h      = heuristic(c.state)
                        self.open_list.put((c.h, c.hash()))


class AstarRec(ReconstructionGoalDetection,Astar):
    pass
class GBFSRec(ReconstructionGoalDetection,GBFS):
    pass

def goalcount(state,goal):
    return np.abs(state-goal).sum()

def blind(state,goal):
    return 0

def main(network_dir, problem_dir, searcher, first_solution=True, heuristics="goalcount", _aae="_aae"):
    global sae, aae, ad, sd3, cae, available_actions
    
    def search(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(searcher, root, ext)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)

    sae = latplan.model.load(network_dir)
    aae = latplan.model.load(sae.local(_aae       ))
    ad  = latplan.model.load(sae.local(_aae+"_ad/"))
    sd3 = latplan.model.load(sae.local(    "_sd3/"))
    cae = latplan.model.load(sae.local(    "_cae/"),allow_failure=True)
    setup_planner_utils(sae, problem_dir, network_dir, "ama2")
    log("loaded sae")

    import importlib
    p = importlib.import_module(sae.parameters["generator"])
    log("loaded puzzle")

    decide_pruning_method()
    
    init, goal = init_goal_misc(p)
    log("loaded init/goal")
    
    known_transisitons = np.loadtxt(sae.local("actions.csv"),dtype=np.int8)
    actions = aae.encode_action(known_transisitons, batch_size=1000).round()
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
    log("initialized actions")

    log("start planning")
    _searcher = eval(searcher)()
    _searcher.stats["aae"]        = _aae
    _searcher.stats["heuristics"] = heuristics
    _searcher.stats["search"]     = searcher
    _searcher.stats["network"]    = network_dir
    _searcher.stats["problem"]    = os.path.normpath(problem_dir).split("/")[-1]
    _searcher.stats["domain"]     = os.path.normpath(problem_dir).split("/")[-2]
    _searcher.stats["noise"]      = os.path.normpath(problem_dir).split("/")[-3]
    _searcher.stats["plan_count"] = 0
    try:
        for i, found_goal_state in enumerate(_searcher.search(init,goal,eval(heuristics))):
            log("plan found")
            _searcher.stats["found"] = True
            _searcher.stats["exhausted"] = False
            _searcher.stats["plan_count"] += 1
            plan = np.array( found_goal_state.path())
            _searcher.stats["statistics"]["cost"] = len(plan)-1
            _searcher.stats["statistics"]["length"] = len(plan)-1
            print(plan)
            if first_solution:
                sae.plot_plan(plan,
                              problem(ama(network(search(heur("problem.png"))))),
                              verbose=True)
            else:
                sae.plot_plan(plan,
                              problem(ama(network(search(heur("problem_{}.png".format(i)))))),
                              verbose=True)
            log("plotted the plan")
            
            validation = p.validate_transitions([sae.decode(plan[0:-1]), sae.decode(plan[1:])])
            print(validation)
            print(ad.discriminate( np.concatenate((plan[0:-1], plan[1:]), axis=-1)).flatten())
            print(p.validate_states(sae.decode(plan)))
            print(combined_sd(plan,sae,cae,sd3).flatten())
            log("validated plan")
            if np.all(validation):
                _searcher.stats["valid"] = True
                return
            _searcher.stats["valid"] = False
            if first_solution:
                return
    except StopIteration:
        _searcher.stats["found"] = False
        _searcher.stats["exhausted"] = True
    finally:
        _searcher.stats["times"] = times
        _searcher.report(problem(ama(network(search(heur("problem.json"))))))

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        sys.exit("{} [networkdir] [problemdir]".format(sys.argv[0]))
    main(*sys.argv[1:])


def test():
    # ./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/
    main("samples/puzzle_mnist33_fc/",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/")
    

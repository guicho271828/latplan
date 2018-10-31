#!/usr/bin/env python3
import warnings
import config_cpu
import numpy as np
import latplan
import latplan.model
from latplan.model import ActionAE, Discriminator, PUDiscriminator
from latplan.util import get_ae_type, bce, mae, mse, ensure_directory
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
combined_discriminator = None

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
    # filtering based on OAE action reconstruction
    action_reconstruction = oae.encode_action(y).round()
    return y[np.all(np.equal(available_actions, action_reconstruction),axis=(1,2))]

def state_reconstruction_from_oae_filtering(y):
    action_reconstruction = oae.encode_action(y)
    N = y.shape[1]//2
    y_reconstruction = oae.decode([y[:,:N], action_reconstruction]).round()
    return y[np.all(np.equal(y, y_reconstruction),axis=(1,))]

def state_reconstruction_filtering(y):
    # filtering based on SAE state reconstruction
    N = y.shape[1]//2
    t = y[:,N:]
    t_reconstruction = sae.autodecode_binary(t).round()
    return y[np.all(np.equal(t, t_reconstruction),axis=(1,))]

def inflate_actions(y):
    from latplan.util import union
    N = y.shape[1]//2
    t = y[:,N:]
    for i in range(inflation-1):
        t = union(sae.autodecode_binary(t).round().astype(int), t)
    import math
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
    return y[np.where(np.squeeze(combined_discriminator(y[:,N:])) > 0.5)[0]]

def cheating_validation_filtering(y):
    N = y.shape[1]//2
    p = latplan.util.puzzle_module(sae.local(""))
    pre_images = sae.decode_binary(y[:,:N],batch_size=1000)
    suc_images = sae.decode_binary(y[:,N:],batch_size=1000)
    return y[p.validate_transitions([pre_images, suc_images],batch_size=1000)]

def cheating_image_reconstruction_filtering(y):
    N = y.shape[1]//2
    # filtering based on SAE reconstruction
    images  = sae.decode_binary(y[:,N:]).round()
    images2 = sae.autoencode(images).round()
    loss = image_diff(images,images2,(1,2))
    # print(loss)
    return y[np.where(loss < image_threshold)].astype(int)

pruning_methods = [
    action_reconstruction_filtering,           # if applied, this should be the first method
    # state_reconstruction_from_oae_filtering,
    inflate_actions,
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
            "generated_including_duplicate":1, # init
            "generated":1,
            "expanded":0,
            "reopened":0,
        }

    def successors(self,state):
        self.stats["expanded"] += 1
        try:
            reductions = []
            s = state.state
            y = oae.decode([np.repeat(np.expand_dims(s,0), len(available_actions), axis=0), available_actions]) \
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
                self.stats["generated_including_duplicate"] += 1
                hash_value = state_hash(succ)
                if hash_value in self.close_list:
                    yield self.close_list[hash_value]
                else:
                    self.stats["generated"] += 1
                    _succ = State(succ)
                    self.close_list[hash_value] = _succ
                    yield _succ
        finally:
            print("->".join(map(str,reductions)))

    def __del__(self):
        print("**************** Search statistics ****************")
        for k, v in self.stats.items():
            print(k, v)

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
                            self.stats["reopened"] += 1
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

def main(network_dir, problem_dir, searcher):
    global sae, oae, ad, ad2, sd, sd2, sd3, cae, combined_discriminator, available_actions
    
    p = latplan.util.puzzle_module(network_dir)

    sae = latplan.model.get(get_ae_type(network_dir))(network_dir).load(allow_failure=True)
    oae = ActionAE(sae.local("_aae/")).load(allow_failure=True)
    try:
        ad  = PUDiscriminator(sae.local("_ad/")).load(allow_failure=True)
    except:
        ad  = Discriminator(sae.local("_ad/")).load(allow_failure=True)
    # sd  = Discriminator(sae.local("_sd/")).load(allow_failure=True)
    # ad2 = Discriminator(sae.local("_ad2/")).load(allow_failure=True)
    # sd2 = Discriminator(sae.local("_sd2/")).load(allow_failure=True)
    sd3 = PUDiscriminator(sae.local("_sd3/")).load()
    try:
        cae = latplan.model.get('SimpleCAE')(sae.local("_cae/")).load()
        combined_discriminator = latplan.model.get('CombinedDiscriminator')(sae,cae,sd3)
    except:
        combined_discriminator = latplan.model.get('CombinedDiscriminator2')(sae,sd3)

    def problem(path):
        return os.path.join(problem_dir,path)
    def network(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(ensure_directory(network_dir).split("/")[-2], root, ext)
    def search(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(searcher, root, ext)


    from scipy import misc
    from latplan.puzzles.util import preprocess, normalize
    # is already enhanced, equalized
    init_image = normalize(misc.imread(problem("init.png")))
    goal_image = normalize(misc.imread(problem("goal.png")))
    print("init:",init_image.min(),init_image.max(),)
    print("goal:",goal_image.min(),goal_image.max(),)
    init = sae.encode_binary(np.expand_dims(init_image,0))[0].round().astype(int)
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0].round().astype(int)
    print(init)
    print(goal)
    rec = sae.decode_binary(np.array([init,goal]))
    init_rec, goal_rec = rec
    print("init (reconstruction):",init_rec.min(),init_rec.max(),)
    print("goal (reconstruction):",goal_rec.min(),goal_rec.max(),)

    def r(i):
        s = i.shape
        return i.reshape((s[0]//2, 2, s[1]//2, 2)).mean(axis=(1,3))
    
    plot_grid([init_image,init_rec,init_image-init_rec,(init_image-init_rec).round(),
               init_image.round(),init_rec.round(),init_image.round()-init_rec.round(),(init_image.round()-init_rec.round()).round(),
               r(init_image),r(init_rec),r(init_image)-r(init_rec),(r(init_image)-r(init_rec)).round(),
               # r(init_image).round(),r(init_rec).round(),r(init_image).round()-r(init_rec).round(),(r(init_image).round()-r(init_rec).round()).round(),
               
               goal_image,goal_rec,goal_image-goal_rec,(goal_image-goal_rec).round(),
               goal_image.round(),goal_rec.round(),goal_image.round()-goal_rec.round(),(goal_image.round()-goal_rec.round()).round(),
               r(goal_image),r(goal_rec),r(goal_image)-r(goal_rec),(r(goal_image)-r(goal_rec)).round(),
               # r(goal_image).round(),r(goal_rec).round(),r(goal_image).round()-r(goal_rec).round(),(r(goal_image).round()-r(goal_rec).round()).round(),
               ],
              w=4,
              path=problem(network("init_goal_reconstruction.png")),verbose=True)

    import sys
    print("init BCE:",bce(init_image,init_rec))
    print("init MAE:",mae(init_image,init_rec))
    print("init MSE:",mse(init_image,init_rec))
    # if image_diff(init_image,init_rec) > image_threshold:
    #     print("Initial state reconstruction failed!")
    #     sys.exit(3)
    print("goal BCE:",bce(goal_image,goal_rec))
    print("goal MAE:",mae(goal_image,goal_rec))
    print("goal MSE:",mse(goal_image,goal_rec))
    # if image_diff(goal_image,goal_rec) > image_threshold:
    #     print("Goal state reconstruction failed!")
    #     sys.exit(4)
    if not np.all(p.validate_states(rec)):
        print("Init/Goal state reconstruction failed!")
        # sys.exit(3)
        print("But we continue anyways...")
    
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

    # available_actions = available_actions.repeat(inflation,axis=0)
        
    for i, found_goal_state in enumerate(eval(searcher)().search(init,goal,goalcount)):
        plan = np.array( found_goal_state.path())
        print(plan)
        plot_grid(sae.decode_binary(plan),
                  path=problem(network(search("path_{}.png".format(i)))),verbose=True)

        validation = p.validate_transitions([sae.decode_binary(plan[0:-1]), sae.decode_binary(plan[1:])])
        print(validation)
        print(ad.discriminate( np.concatenate((plan[0:-1], plan[1:]), axis=-1)).flatten())

        print(p.validate_states(sae.decode_binary(plan)))
        print(combined_discriminator(plan).flatten())
        import subprocess
        subprocess.call(["rm", "-f", problem(network(search("path_{}.valid".format(i))))])
        if np.all(validation):
            subprocess.call(["touch", problem(network(search("path_{}.valid".format(i))))])
            sys.exit(0)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        sys.exit("{} [networkdir] [problemdir]".format(sys.argv[0]))
    main(*sys.argv[1:])


def test():
    # ./trivial-planner.py samples/puzzle_mnist33_fc/ trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/
    main("samples/puzzle_mnist33_fc/",
         "trivial-planner-instances/latplan.puzzles.puzzle_mnist/0-0/")
    

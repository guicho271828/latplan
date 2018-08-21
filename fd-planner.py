#!/usr/bin/env python3

import config_cpu
import numpy as np
import subprocess
import os
import latplan
from latplan.model import default_networks
from latplan.util import get_ae_type, bce, mae, mse, ensure_directory
from latplan.util.plot import plot_grid
import os.path
import keras.backend as K
import tensorflow as tf
import math

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def echodo(cmd,*args,**kwargs):
    print(cmd,flush=True)
    subprocess.call(cmd,*args,**kwargs)

def echo_out(cmd):
    print(cmd)
    return subprocess.check_output(cmd)

class PlanException(BaseException):
    pass

sae = None
problem_dir = None
network_dir = None

options = {
    "lmcut" : "--search astar(lmcut())",
    "blind" : "--search astar(blind())",
    "hmax"  : "--search astar(hmax())",
    "mands" : "--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(max_states=50000,greedy=false),merge_strategy=merge_dfp(),label_reduction=exact(before_shrinking=true,before_merging=false)))",
    "pdb"   : "--search astar(pdb())",
    "cpdb"  : "--search astar(cpdbs())",
    "ipdb"  : "--search astar(ipdb())",
    "zopdb"  : "--search astar(zopdbs())",
}

option = "blind"
action_type = "all_actions"

def problem(path):
    return os.path.join(problem_dir,path)

def network(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ensure_directory(network_dir).split("/")[-2], root, ext)

def preprocess(bits):
    np.savetxt(problem(network("ama1_ig.csv")),[bits],"%d")
    echodo(["touch",problem("domain.pddl")]) # dummy file, just making planner-scripts work
    echodo(["helper/sas.sh",
            os.path.join(ensure_directory(network_dir),"{}.csv".format(action_type)),
            problem(network("ama1_ig.csv")),
            problem(network("{}.sas.gz".format(action_type)))])
    echodo(["helper/sasp.sh",
            problem(network("{}.sas.gz".format(action_type))),
            problem(network("{}.sasp.gz".format(action_type)))])

def latent_plan(init,goal,mode):
    bits = np.concatenate((init,goal))
    ###### preprocessing ################################################################

    ## old code for caching...
    # lock = problem(network("lock"))
    # import fcntl
    # try:
    #     with open(lock) as f:
    #         print("lockfile found!")
    #         fcntl.flock(f, fcntl.LOCK_SH)
    # except FileNotFoundError:
    #     with open(lock,'wb') as f:
    #         fcntl.flock(f, fcntl.LOCK_EX)
    #         preprocess(bits)
            
    preprocess(bits)
    
    ###### do planning #############################################
    sasp     = problem(network("{}.sasp.gz".format(action_type)))
    plan_raw = problem(network("{}.sasp.gz.plan".format(action_type)))
    plan     = problem(network("{}.{}.plan".format(action_type,mode)))
    
    echodo(["helper/fd-sasgz.sh",options[mode], sasp])
    assert os.path.exists(plan_raw)
    echodo(["mv",plan_raw,plan])
    
    out = echo_out(["lisp/parse-plan.bin",plan, *list(init.astype('str'))])
    lines = out.splitlines()
    return np.array([ [ int(s) for s in l.split() ] for l in lines ])

def init_goal_misc(p, init_image, goal_image, init, goal):
    print("init:",init_image.min(),init_image.max(),)
    print("goal:",goal_image.min(),goal_image.max(),)
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

def main(_network_dir, _problem_dir, heuristics='blind'):
    global sae, problem_dir, network_dir
    problem_dir = _problem_dir
    network_dir = _network_dir
    p = latplan.util.puzzle_module(network_dir)
    sae = default_networks[get_ae_type(network_dir)](network_dir).load(allow_failure=True)

    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    
    from scipy import misc
    from latplan.puzzles.util import preprocess, normalize
    # is already enhanced, equalized
    init_image = normalize(misc.imread(problem("init.png")))
    goal_image = normalize(misc.imread(problem("goal.png")))
    init = sae.encode_binary(np.expand_dims(init_image,0))[0].round().astype(int)
    goal = sae.encode_binary(np.expand_dims(goal_image,0))[0].round().astype(int)
    init_goal_misc(p, init_image, goal_image, init, goal)
    plan = latent_plan(init, goal, heuristics)
    print(plan)
    plot_grid(sae.decode_binary(plan),
              path=problem(network(heur("path_{}.png".format(0)))),verbose=True)

    validation = p.validate_transitions([sae.decode_binary(plan[0:-1]), sae.decode_binary(plan[1:])])
    print(validation)
    print(p.validate_states(sae.decode_binary(plan)))
    
    import subprocess
    subprocess.call(["rm", "-f", problem(network(heur("path_{}.valid".format(0))))])
    if np.all(validation):
        subprocess.call(["touch", problem(network(heur("path_{}.valid".format(0))))])
        sys.exit(0)





import sys
print(sys.argv)
main(*sys.argv[1:])

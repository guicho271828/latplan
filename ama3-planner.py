#!/usr/bin/env python3

import config_cpu
import numpy as np
import subprocess
import os
import sys
import latplan
import latplan.model
from latplan.util import get_ae_type, bce, mae, mse, ensure_directory
from latplan.util.plot import plot_grid
import os.path
import keras.backend as K
import tensorflow as tf
import math
import time
import json

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

start = time.time()
times = [(0,0,"init")]
def log(message):
    now = time.time()
    wall = now-start
    elap = wall-times[-1][0]
    times.append((wall,elap,message))
    print("@[{: =10.3f} +{: =10.3f}] {}".format(wall,elap,message))

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
    "ff"    : "--search eager(single(ff()))",
    "mands" : "--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(max_states=50000,greedy=false),merge_strategy=merge_dfp(),label_reduction=exact(before_shrinking=true,before_merging=false)))",
    "pdb"   : "--search astar(pdb())",
    "cpdb"  : "--search astar(cpdbs())",
    "ipdb"  : "--search astar(ipdb())",
    "zopdb"  : "--search astar(zopdbs())",
}

def problem(path):
    return os.path.join(problem_dir,path)

def network(path):
    root, ext = os.path.splitext(path)
    return "{}_{}{}".format(ensure_directory(network_dir).split("/")[-2], root, ext)

def init_goal_misc(p, init_image, goal_image, init, goal):
    print("init:",init_image.min(),init_image.max(),)
    print("goal:",goal_image.min(),goal_image.max(),)
    print(init)
    print(goal)
    rec = sae.decode(np.array([init,goal]))
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

def main(domainfile, _problem_dir, heuristics):

    log("main")
    global sae, problem_dir, network_dir

    problem_dir = _problem_dir
    network_dir = os.path.dirname(domainfile)
    print(problem_dir,network_dir)
    p = latplan.util.puzzle_module(network_dir)
    log("loaded puzzle")
    sae = latplan.model.get(get_ae_type(network_dir))(network_dir).load(allow_failure=True)
    log("loaded sae")

    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    
    import imageio
    from latplan.puzzles.util import preprocess, normalize
    # is already enhanced, equalized
    init_image = normalize(imageio.imread(problem("init.png")))
    goal_image = normalize(imageio.imread(problem("goal.png")))
    init = sae.encode(np.expand_dims(init_image,0))[0].round().astype(int)
    goal = sae.encode(np.expand_dims(goal_image,0))[0].round().astype(int)
    log("loaded init/goal")
    init_goal_misc(p, init_image, goal_image, init, goal)
    log("visualized init/goal")

    bits = np.concatenate((init,goal))
    ###### preprocessing ################################################################
    np.savetxt(problem(network("ama1_ig.csv")),[bits],"%d")

    def output(ext):
        return problem(network("{}_{}.{}".format(os.path.splitext(os.path.basename(domainfile))[0], heuristics, ext)))

    echodo(["helper/problem.sh",
            problem(network("ama1_ig.csv")),
            output("pddl")])
    log("generated problem")
    
    ###### do planning #############################################
    echodo(["helper/fd.sh", options[heuristics], output("pddl"), domainfile])
    log("finished planning")
    assert os.path.exists(output("plan"))

    echodo(["arrival", domainfile, output("pddl"), output("plan"), output("trace")])
        
    log("simulated the plan")
    echodo(["lisp/read-latent-state-traces.bin", output("trace"), str(len(init)), output("csv")])
    plan = np.loadtxt(output("csv"), dtype=int)
    log("parsed the plan")
    plot_grid(sae.decode(plan), path=output("png"), verbose=True)
    log("plotted the plan")
    validation = p.validate_transitions([sae.decode(plan[0:-1]), sae.decode(plan[1:])])
    print(validation)
    print(p.validate_states(sae.decode(plan)))
    log("validated the plan")
    with open(output("json"),"w") as f:
        json.dump({
            "times":times,
            "heuristics":heuristics,
            "domainfile":domainfile,
            "parameters":sae.parameters,
            "valid":bool(np.all(validation)),
        }, f)
    
    import subprocess
    if np.all(validation):
        subprocess.call(["touch", output("valid")])


import sys
print(sys.argv)
main(*sys.argv[1:])

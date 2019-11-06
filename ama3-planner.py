#!/usr/bin/env python3

import config_cpu
import numpy as np
import subprocess
import os
import sys
import latplan
import latplan.model
from latplan.util import *
from latplan.util.planner import *
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

class PlanException(BaseException):
    pass

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

def main(domainfile, problem_dir, heuristics):
    network_dir = os.path.dirname(domainfile)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    
    p = latplan.util.puzzle_module(network_dir)
    log("loaded puzzle")
    sae = latplan.model.load(network_dir,allow_failure=True)
    log("loaded sae")
    setup_planner_utils(sae, problem_dir, network_dir, "ama3")

    init, goal = init_goal_misc(p)
    log("loaded init/goal")

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
            "network":network_dir,
            "problem":problem_dir,
            "times":times,
            "heuristics":heuristics,
            "domainfile":domainfile,
            "parameters":sae.parameters,
            "valid":bool(np.all(validation)),
        }, f)
    
import sys
print(sys.argv)
main(*sys.argv[1:])

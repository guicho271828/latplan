#!/usr/bin/env python3

import config_cpu
import numpy as np
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
    "mands" : "--search astar(merge_and_shrink(shrink_strategy=shrink_bisimulation(max_states=50000,greedy=false),merge_strategy=merge_dfp(),label_reduction=exact(before_shrinking=true,before_merging=false)))",
    "pdb"   : "--search astar(pdb())",
    "cpdb"  : "--search astar(cpdbs())",
    "ipdb"  : "--search astar(ipdb())",
    "zopdb"  : "--search astar(zopdbs())",
}

def main(network_dir, problem_dir, heuristics='blind', action_type="all_actions"):

    p = latplan.util.puzzle_module(network_dir)
    log("loaded puzzle")
    sae = latplan.model.load(network_dir,allow_failure=True)
    log("loaded sae")
    setup_planner_utils(sae, problem_dir, network_dir, "ama1")

    init, goal = init_goal_misc(p)
    log("loaded init/goal")
    log("start planning")

    bits = np.concatenate((init,goal))
    ###### preprocessing ################################################################

    np.savetxt(problem(network("{}_{}.csv".format(action_type,heuristics))),[bits],"%d")
    echodo(["touch",problem("domain.pddl")]) # dummy file, just making planner-scripts work
    echodo(["helper/sas.sh",
            os.path.join(ensure_directory(network_dir),"{}.csv".format(action_type)),
            problem(network("{}_{}.csv".format(action_type,heuristics))),
            problem(network("{}.sas.gz".format(action_type)))])
    log("sas.sh done")
    echodo(["helper/sasp.sh",
            problem(network("{}.sas.gz".format(action_type))),
            problem(network("{}.sasp.gz".format(action_type)))])
    log("sasp.sh done")
    
    ###### do planning #############################################
    sasp     = problem(network("{}.sasp.gz".format(action_type)))
    plan_raw = problem(network("{}.sasp.gz.plan".format(action_type)))
    planfile = problem(network("{}_{}.plan".format(action_type,heuristics)))
    
    echodo(["helper/fd-sasgz.sh",options[heuristics], sasp])
    log("fd-sasgz.sh done")
    assert os.path.exists(plan_raw)
    echodo(["mv",plan_raw,planfile])
    
    ###### parse the plan #############################################
    out = echo_out(["lisp/parse-plan.bin",planfile, *list(init.astype('str'))])
    lines = out.splitlines()
    plan = np.array([ [ int(s) for s in l.split() ] for l in lines ])
    log("parsed the plan")
    plot_grid(sae.decode(plan),
              path=problem(network("{}_{}.png".format(action_type,heuristics))),
              verbose=True)
    log("plotted the plan")

    validation = p.validate_transitions([sae.decode(plan[0:-1]), sae.decode(plan[1:])])
    print(validation)
    print(p.validate_states(sae.decode(plan)))
    log("validated plan")
    with open(problem(network("{}_{}.json".format(action_type,heuristics))),"w") as f:
        json.dump({
            "network":network_dir,
            "problem":problem_dir,
            "times":times,
            "heuristics":heuristics,
            "action_type":action_type,
            "parameters":sae.parameters,
            "valid":bool(np.all(validation)),
        }, f)
    
import sys
print(sys.argv)
main(*sys.argv[1:])

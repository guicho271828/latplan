#!/usr/bin/env python3

import numpy as np
import os
import sys
import latplan
import latplan.model
from latplan.util import *
from latplan.util.planner import *
from latplan.util.plot import *
import os.path
import keras.backend as K
import tensorflow as tf
import json

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

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

    def action(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(action_type, root, ext)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)

    sae = latplan.model.load(network_dir,allow_failure=True)
    log("loaded sae")
    import importlib
    p = importlib.import_module(sae.parameters["generator"])
    log("loaded puzzle")
    setup_planner_utils(sae, problem_dir, network_dir, "ama1")

    init, goal = init_goal_misc(p)
    log("loaded init/goal")
    log("start planning")

    bits = np.concatenate((init,goal))

    ###### files ################################################################
    transitions = sae.local("{}.csv".format(action_type))
    ig       = problem(ama(network("ig.csv")))
    sas      = problem(ama(network(action("sas.gz"))))
    sasp     = problem(ama(network(action("sasp.gz"))))
    sasp2    = problem(ama(network(action(heur("sasp.gz")))))
    planfile = problem(ama(network(action(heur("sasp.gz.plan")))))
    pngfile  = problem(ama(network(action(heur("plan.png")))))
    jsonfile = problem(ama(network(action(heur("plan.json")))))
    
    ###### preprocessing ################################################################
    os.path.exists(ig) or np.savetxt(ig,[bits],"%d")
    echodo(["touch",problem("domain.pddl")]) # dummy file, just making planner-scripts work
    echodo(["helper/ama1-sas.sh", transitions, ig, sas])
    log("sas.sh done")
    echodo(["helper/ama1-sasp.sh", sas, sasp, sasp2])
    log("sasp.sh done")
    
    ###### do planning #############################################
    
    echodo(["helper/fd-sasgz.sh",options[heuristics], sasp2])
    log("fd-sasgz.sh done")
    assert os.path.exists(planfile)
    
    ###### parse the plan #############################################
    out = echo_out(["lisp/ama1-parse-plan.bin",planfile, *list(init.astype('str'))])
    lines = out.splitlines()
    plan = np.array([ [ int(s) for s in l.split() ] for l in lines ])
    log("parsed the plan")
    sae.plot_plan(plan, pngfile, verbose=True)
    log("plotted the plan")

    validation = p.validate_transitions([sae.decode(plan[0:-1]), sae.decode(plan[1:])])
    print(validation)
    print(p.validate_states(sae.decode(plan)))
    log("validated plan")
    with open(jsonfile,"w") as f:
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

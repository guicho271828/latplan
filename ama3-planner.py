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
from latplan.util.plot import *
import latplan.util.stacktrace
import os.path
import keras.backend as K
import tensorflow as tf
import math
import time
import json

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})


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
    network_dir = domainfile
    success = False
    while not success:
        network_dir = os.path.dirname(network_dir)
        print(network_dir)
        if network_dir == "":
            break
        try:
            p = puzzle_module(network_dir)
            success = True
        except Exception as e:
            print(e)
            latplan.util.stacktrace.format(False)
    if not success:
        sys.exit("could not locate the network")
    domainfile_rel = os.path.relpath(domainfile, network_dir)
    
    def domain(path):
        dom_prefix = domainfile_rel.replace("/","_")
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(os.path.splitext(dom_prefix)[0], root, ext)
    def heur(path):
        root, ext = os.path.splitext(path)
        return "{}_{}{}".format(heuristics, root, ext)
    
    log("loaded puzzle")
    sae = latplan.model.load(network_dir,allow_failure=True)
    log("loaded sae")
    setup_planner_utils(sae, problem_dir, network_dir, "ama3")

    init, goal = init_goal_misc(p)
    log("loaded init/goal")

    log("start planning")
    
    bits = np.concatenate((init,goal))

    ###### files ################################################################
    ig          = problem(ama(network("ig.csv")))
    problemfile = problem(ama(network(domain(heur("problem.pddl")))))
    planfile    = problem(ama(network(domain(heur("problem.plan")))))
    tracefile   = problem(ama(network(domain(heur("problem.trace")))))
    csvfile     = problem(ama(network(domain(heur("problem.csv")))))
    pngfile     = problem(ama(network(domain(heur("problem.png")))))
    jsonfile    = problem(ama(network(domain(heur("problem.json")))))
    
    ###### preprocessing ################################################################
    os.path.exists(ig) or np.savetxt(ig,[bits],"%d")
    echodo(["helper/problem.sh",ig,problemfile])
    log("generated problem")
    
    ###### do planning #############################################
    echodo(["helper/fd.sh", options[heuristics], problemfile, domainfile])
    log("finished planning")
    assert os.path.exists(planfile)

    log("running a validator")
    echodo(["arrival", domainfile, problemfile, planfile, tracefile])
        
    log("simulated the plan")
    echodo(["lisp/read-latent-state-traces.bin", tracefile, str(len(init)), csvfile])
    plan = np.loadtxt(csvfile, dtype=int)
    log("parsed the plan")
    plot_grid(sae.decode(plan), path=pngfile, verbose=True)
    log("plotted the plan")
    validation = p.validate_transitions([sae.decode(plan[0:-1]), sae.decode(plan[1:])])
    print(validation)
    print(p.validate_states(sae.decode(plan)))
    log("validated the plan")
    with open(jsonfile,"w") as f:
        json.dump({
            "network":network_dir,
            "problem":os.path.normpath(problem_dir).split("/")[-1],
            "domain" :os.path.normpath(problem_dir).split("/")[-2],
            "noise"  :os.path.normpath(problem_dir).split("/")[-3],
            "times":times,
            "heuristics":heuristics,
            "domainfile":domainfile.split("/"),
            "problemfile":problemfile,
            "planfile":planfile,
            "tracefile":tracefile,
            "csvfile":csvfile,
            "pngfile":pngfile,
            "jsonfile":jsonfile,
            "parameters":sae.parameters,
            "cost":len(plan),
            "valid":bool(np.all(validation)),
        }, f)
    
import sys
print(sys.argv)
main(*sys.argv[1:])

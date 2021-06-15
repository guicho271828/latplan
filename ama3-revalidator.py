#!/usr/bin/env python3

# This script revalidates an existing result without planning.

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

def main(domainfile, problem_dir, heuristics):
    network_dir = os.path.dirname(domainfile)
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

    p = puzzle_module(sae)
    log("loaded puzzle")

    ###### files ################################################################
    pngfile     = problem(ama(network(domain(heur(f"problem.png")))))
    jsonfile    = problem(ama(network(domain(heur(f"problem.json")))))
    npzfile     = problem(ama(network(domain(heur(f"problem.npz")))))

    log(f"start loading the plan")
    with np.load(npzfile) as data:
        plan_images = data["img_states"]
    log(f"finished archiving the plan")

    log(f"start visually validating the plan image : transitions")
    # note: only puzzle, hanoi, lightsout have the custom validator, which are all monochrome.
    plan_images = sae.render(plan_images) # unnormalize the image
    validation = p.validate_transitions([plan_images[0:-1], plan_images[1:]])
    print(validation)
    valid = bool(np.all(validation))
    log(f"finished visually validating the plan image : transitions")

    log(f"start visually validating the plan image : states")
    print(p.validate_states(plan_images))
    log(f"finished visually validating the plan image : states")

    with open(jsonfile,"r") as f:
        data = json.load(f)

    print(f"previously, this solution was {data['valid']}. Now it is {valid}")
    data["valid"] = valid
    with open(jsonfile,"w") as f:
        json.dump(data, f, indent=2)
    return valid


if __name__ == '__main__':
    try:
        main(*sys.argv[1:])
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()

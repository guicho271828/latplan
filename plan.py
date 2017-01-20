#!/usr/bin/env python3

import config
import numpy as np
import subprocess
import os
from plot import plot_grid, plot_grid2, plot_ae

def echodo(cmd,file=None):
    subprocess.call(["echo"]+cmd)
    if file is None:
        subprocess.call(cmd)
    else:
        with open(file,"w") as f:
            subprocess.call(cmd,stdout=f)

class PlanException(BaseException):
    pass

def latent_plan(init,goal,ae,use_augmented=False):
    ig_x, ig_z, ig_y, ig_b, ig_by = plot_ae(ae,np.array([init,goal]),"init_goal.png")

    # start planning
    
    echodo(["make","-C","lisp","-j","1"])
    echodo(["rm",ae.local("problem.plan")])
    if use_augmented:
        if not os.path.exists(ae.local("domain.pddl")) or \
           os.path.getmtime(ae.local("augmented.csv")) > \
           os.path.getmtime(ae.local("domain.pddl")):
            echodo(["lisp/domain.bin",ae.local("augmented.csv")],
                   ae.local("domain.pddl"))
        else:
            print("skipped generating domain.pddl")
    else:
        if not os.path.exists(ae.local("domain.pddl")) or \
           os.path.getmtime(ae.local("actions.csv")) > \
           os.path.getmtime(ae.local("domain.pddl")):
            echodo(["lisp/domain.bin",ae.local("actions.csv")],
                   ae.local("domain.pddl"))
        else:
            print("skipped generating domain.pddl")
    echodo(["lisp/problem.bin",
            *list(ig_b.flatten().astype('int').astype('str'))],
           ae.local("problem.pddl"))
    echodo(["planner-scripts/limit.sh","-v","-t","30",
            "-o","--alias lama-first","--","fd-alias-clean",
            ae.local("problem.pddl"),
            ae.local("domain.pddl")])
    if not os.path.exists(ae.local("problem.plan")):
        raise PlanException("no plan found")
    subprocess.call(["echo"]+["lisp/parse-plan.bin",ae.local("problem.plan"),
                              *list(ig_b[0].flatten().astype('int').astype('str'))])
    out = subprocess.check_output(["lisp/parse-plan.bin",ae.local("problem.plan"),
                                   *list(ig_b[0].flatten().astype('int').astype('str'))])
    lines = out.splitlines()
    if len(lines) is 2:
        raise PlanException("not an interesting problem")
    numbers = np.array([ [ int(s) for s in l.split() ] for l in lines ])
    print(numbers)
    plan_images = ae.decode_binary(numbers)
    plot_grid(plan_images,path=ae.local('plan.png'))

def select(data,num):
    return data[np.random.randint(0,data.shape[0],num)]

if __name__ == '__main__':
    import random
    from model import GumbelAE
    ae = GumbelAE("samples/mnist_puzzle33p_model/")
    import mnist_puzzle
    configs = np.array(list(mnist_puzzle.generate_configs(9)))
    while True:
        ig_c = select(configs,2)
        ig = mnist_puzzle.states(3,3,ig_c)
        try:
            latent_plan(*ig, ae, use_augmented=True)
            break
        except PlanException as e:
            print(e)
    print("The problem was solvable. Trying the original formulation")
    latent_plan(*ig, ae, use_augmented=False)
    print("Original formulation is also solvable.")
    
    
    

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

def latent_plan(init,goal,ae,mode='original'):
    ig_x, ig_z, ig_y, ig_b, ig_by = plot_ae(ae,np.array([init,goal]),"init_goal.png")

    def original():
        d = ae.local("domain.pddl")
        a = ae.local("actions.csv")
        if not os.path.exists(d) or os.path.getmtime(a) > os.path.getmtime(d):
            echodo(["lisp/domain.bin",a], d)
        else:
            print("skipped generating {}".format(d))
        return d
    def augmented():
        d = ae.local("augmented.pddl")
        a = ae.local("augmented.csv")
        if not os.path.exists(d) or os.path.getmtime(a) > os.path.getmtime(d):
            echodo(["lisp/domain.bin",a], d)
        else:
            print("skipped generating {}".format(d))
        return d
    def msdd():
        d = ae.local("msdd.pddl")
        a = ae.local("augmented.csv")
        if not os.path.exists(d) or os.path.getmtime(a) > os.path.getmtime(d):
            echodo(["lisp/msdd.ros", "-t", "-k", "200", "-n", "1000", a], d)
        else:
            print("skipped generating {}".format(d))
        return d
    
    # start planning

    echodo(["make","-C","lisp","-j","1"])
    echodo(["rm",ae.local("problem.plan")])
    
    domain = locals()[mode]()
    echodo(["lisp/problem.bin",
            *list(ig_b.flatten().astype('int').astype('str'))],
           ae.local("problem.pddl"))
    echodo(["planner-scripts/limit.sh","-v","-t","30",
            "--","ff-clean",
            ae.local("problem.pddl"),
            domain])
    if os.path.exists(ae.local("problem.negative")):
        echodo(["rm",ae.local("problem.negative")])
        raise PlanException("goal can be simplified to FALSE. No plan will solve it")
    if not os.path.exists(ae.local("problem.plan")):
        echodo(["planner-scripts/limit.sh","-v","-t","30",
                "-o","--alias lama-first","--","fd-alias-clean",
                ae.local("problem.pddl"),
                domain])
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
            latent_plan(*ig, ae)
            break
        except PlanException as e:
            print(e)
    # print("The problem was solvable. Trying the original formulation")
    # latent_plan(*ig, ae)
    # print("Original formulation is also solvable.")
    
    
    

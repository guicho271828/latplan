#!/usr/bin/env python3

import numpy as np
import subprocess

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
    if use_augmented:
        echodo(["lisp/domain.bin",ae.local("augmented.csv")],
               ae.local("domain.pddl"))
    else:
        echodo(["lisp/domain.bin",ae.local("actions.csv")],
               ae.local("domain.pddl"))
    echodo(["lisp/problem.bin",
            *list(ig_b.flatten().astype('int').astype('str'))],
           ae.local("problem.pddl"))
    echodo(["planner-scripts/limit.sh","-v","--","fd-clean",
            ae.local("problem.pddl"),
            ae.local("domain.pddl")])
    try:
        out = subprocess.check_output(["lisp/parse-plan.bin",ae.local("problem.plan"),
                                       *list(ig_b[0].flatten().astype('int').astype('str'))])
        lines = out.splitlines()
        if len(lines) is 2:
            raise PlanException("not an interesting problem")
        numbers = np.array([ [ int(s) for s in l.split() ] for l in lines ])
        print(numbers)
        latent_dim = numbers.shape[1]/2
        states = np.concatenate((numbers[0:1,0:latent_dim],
                                 numbers[:,latent_dim:]))
        print(states)
        plan_images = ae.decode_binary(states)
        plot_grid(plan_images,path=ae.local('plan.png'))
    except subprocess.CalledProcessError:
        raise PlanException("no plan found")

def select(data,num):
    return data[np.random.randint(0,data.shape[0],num)]

if __name__ == '__main__':
    import random
    from model import GumbelAE
    ae = GumbelAE("samples/mnist_puzzle33p_model/")
    import mnist_puzzle
    configs = np.array(list(mnist_puzzle.generate_configs(9)))
    ig_c = select(configs,2)
    ig = mnist_puzzle.states(3,3,ig_c)
    while True:
        try:
            latent_plan(*ig, ae, use_augmented=True)
            break
        except PlanException as e:
            print(e)
    
    
    

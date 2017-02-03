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

def echo_out(cmd):
    subprocess.call(["echo"]+cmd)
    return subprocess.check_output(cmd)

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

def latent_plan(init,goal,ae,mode='lmcut'):
    ig_x, ig_z, ig_y, ig_b, ig_by = plot_ae(ae,np.array([init,goal]),"init_goal.png")

    # np.savetxt(ae.local("problem.csv"),ig_b.flatten().astype('int'),"%d")

    # start planning
    plan_raw = ae.local("problem.sasp.plan")
    plan = ae.local("{}.plan".format(mode))
    echodo(["rm",plan])
    echodo(["make","-C","lisp","-j","1"])
    echodo(["make","-C",ae.path,"-f","../Makefile"])
    echodo(["planner-scripts/limit.sh","-v","-t","3600",
            "-o",options[mode],
            "--","fd-sas-clean",
            ae.local("problem.sasp")])
    if not os.path.exists(plan_raw):
        raise PlanException("no plan found")
    echodo(["mv",plan_raw,plan])
    out = echo_out(["lisp/parse-plan.bin",plan,
                    *list(ig_b[0].flatten().astype('int').astype('str'))])
    lines = out.splitlines()
    if len(lines) is 2:
        raise PlanException("not an interesting problem")
    numbers = np.array([ [ int(s) for s in l.split() ] for l in lines ])
    print(numbers)
    plan_images = ae.decode_binary(numbers)
    plot_grid(plan_images,path=ae.local('{}.png'.format(mode)))

def select(data,num):
    return data[np.random.randint(0,data.shape[0],num)]

def run_puzzle(path, p):
    from model import GumbelAE
    ae = GumbelAE(path)
    configs = np.array(list(p.generate_configs(9)))
    def convert(panels):
        return np.array([
            [i for i,x in enumerate(panels) if x == p]
            for p in range(9)]).reshape(-1)
    ig_c = [convert([8,0,6,5,4,7,2,3,1]),
            convert([0,1,2,3,4,5,6,7,8])]
    ig = p.states(3,3,ig_c)
    try:
        latent_plan(*ig, ae, sys.argv[1])
    except PlanException as e:
        print(e)
    
def run_lightsout(path, p):
    from model import GumbelAE
    ae = GumbelAE(path)
    configs = np.array(list(p.generate_configs(3)))
    ig_c = [[0,0,1,
             0,1,0,
             1,0,1,],
            np.zeros(9)]
    ig = p.states(3,ig_c)
    try:
        latent_plan(*ig, ae, sys.argv[1])
    except PlanException as e:
        print(e)
    

if __name__ == '__main__':
    import sys
    import random
    # import puzzles.mnist_puzzle as p
    # run_puzzle("samples/mnist_puzzle33p_model/",p)
    import puzzles.lenna_puzzle as p
    run_puzzle("samples/lenna_puzzle33p_model/",p)
    # import puzzles.spider_puzzle as p
    # run_puzzle("samples/spider_puzzle33p_model/",p)
    # import puzzles.digital_lightsout as p
    # run_lightsout("samples/digital_lightsout_model/",p)
    

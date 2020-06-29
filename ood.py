#!/usr/bin/env python3

import numpy as np
import latplan

import os.path
import json

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

################################################################

def count_appearance(configs1,configs2):
    count = 0
    for config1 in configs1:
        for config2 in configs2:
            if np.all(config1 == config2):
                count+=1
                break
    return count

def common_operation(path,p,num_examples,plannpz,planjson,allow_invalid,*args):
    with open(planjson, 'r') as f:
        data = json.load(f)
    if not allow_invalid and not data["valid"]:
        print("this plan is not valid")
        return
    
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]
    with np.load(plannpz) as data:
        img_states = data['img_states']
    
    results = {}
    
    plan_configs = p.to_configs(img_states)

    # states that have never been used

    print("plan length")
    results["s0"] = len(plan_configs)
    print("states that are found in the input dataset")
    results["s1"] = count_appearance(plan_configs, np.concatenate((pre_configs,suc_configs),axis=0))

    print("transitions that are found in the input dataset")
    conc_configs = np.concatenate([pre_configs,suc_configs],axis=-1)

    pre_configs_plan = plan_configs[0:-1]
    suc_configs_plan = plan_configs[1:]
    conc_configs_plan = np.concatenate([pre_configs_plan,suc_configs_plan],axis=-1)

    results["t"] = count_appearance(conc_configs_plan, conc_configs)

    basename, _ = os.path.splitext(plannpz)
    out = basename+"-ood.json"
    print("writing results to", out)
    with open(out,"w") as f:
        json.dump(results,f)

def puzzle(type='mnist',width=3,height=3,num_examples=6500,plannpz=None,planjson=None,allow_invalid=False):
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()

    common_operation(path, p, num_examples, plannpz, planjson, allow_invalid, width, height)

def hanoi(disks=7,towers=4,num_examples=6500,plannpz=None,planjson=None,allow_invalid=False):
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")

    import latplan.puzzles.hanoi as p
    p.setup()

    common_operation(path, p, num_examples, plannpz, planjson, allow_invalid, disks, towers)

def lightsout(type='digital',size=4,num_examples=6500,plannpz=None,planjson=None,allow_invalid=False):
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["lightsout",type,size]))+".npz")
    
    import importlib
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    
    common_operation(path, p, num_examples, plannpz, planjson, allow_invalid)

def main():
    import sys
    if len(sys.argv) == 1:
        print({ k for k in dir(latplan.model)})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        task = sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()


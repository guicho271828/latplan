#!/usr/bin/env python3

import config
import numpy as np
import random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import grid_search, nn_task
from latplan.util.noise  import gaussian

import keras.backend as K
import tensorflow as tf

import os
import os.path

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

################################################################

def puzzle(type='mnist',width=3,height=3,limit=None):
    # limit = number that "this much is enough"
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    pres = p.generate_random_configs(width*height, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1,width,height)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)

def hanoi(disks=7,towers=4,limit=None):
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    import latplan.puzzles.hanoi as p
    p.setup()
    pres = p.generate_random_configs(disks,towers, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1,disks,towers)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)

def lightsout(type='digital',size=4,limit=None):
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["lightsout",type,size]))+".npz")
    import importlib
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    pres = p.generate_random_configs(size, limit)
    np.random.shuffle(pres)
    sucs = [ random.choice(p.successors(c1)) for c1 in pres ]
    np.savez_compressed(path,pres=pres,sucs=sucs)

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
    main()

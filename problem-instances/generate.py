#!/usr/bin/env python3

import numpy as np
import os
import os.path
import sys
sys.path.append('../../')
from latplan.util import curry
from latplan.util.noise import gaussian, salt, pepper, saltpepper

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import keras.backend as K
tf.logging.set_verbosity(tf.logging.ERROR)
# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))

def load_session():
    K.set_session(
        tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options =
                tf.GPUOptions(
                    per_process_gpu_memory_fraction=1.0,
                    allow_growth=True,))))

load_session()

def identity(x):
    return x

steps     = 5
instances = 100
noise_fns     = [identity]
output_dirs = ["vanilla"]

def random_walk(init_c,length,successor_fn):
    print(".",end="")
    while True:
        result = random_walk_rec(init_c, [init_c], length, successor_fn)
        print()
        if result is None:
            continue
        else:
            return result

def random_walk_rec(current, trace, length, successor_fn): 
    import numpy.random as random
    if length == 0:
        return current
    else:
        sucs = successor_fn(current)
        first = random.randint(len(sucs))
        now = first

        while True:
            suc = sucs[now]
            try:
                assert not np.any([np.all(np.equal(suc, t)) for t in trace])
                result = random_walk_rec(suc, [*trace, suc], length-1, successor_fn)
                assert result is not None
                return result
            except AssertionError:
                now = (now+1)%len(sucs)
                if now == first:
                    print("B",end="")
                    return None
                else:
                    continue

def safe_chdir(path):
    try:
        os.mkdir(path)
    except:
        pass
    os.chdir(path)

def generate(p, ics, gcs, *args):
    from scipy import misc
    import subprocess
    import datetime
    inits = p.generate(np.array(ics),*args)
    goals = p.generate(np.array(gcs),*args)
    for noise_fn,output_dir in zip(noise_fns,output_dirs):
        inits = noise_fn(inits)
        goals = noise_fn(goals)
        for i,init in enumerate(inits):
            for j,goal in enumerate(goals):
                d = "{}/{}/{:03d}-{:03d}-{:03d}".format(output_dir,p.__name__,steps,i,j)
                try:
                    subprocess.call(["mv",d,d+"_old_"+datetime.datetime.today().isoformat()])
                except:
                    pass
                os.makedirs(d)
                print(d)
                misc.imsave(os.path.join(d,"init.png"),init)
                misc.imsave(os.path.join(d,"goal.png"),goal)

################################################################

def puzzle(type='mnist', width=3, height=3):
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    ics = [
        random_walk(np.arange(width*height), steps, lambda config: p.successors(config,width,height))
        for i in range(instances)
    ]
    gcs = np.arange(width*height).reshape((1,width*height))
    generate(p, ics, gcs, width, height)

def puzzle_longest(type='mnist', width=3, height=3):
    global output_dirs
    output_dirs = ["longest"]
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    ics = [
        # from Reinfield '93
        # [8,0,6,5,4,7,2,3,1], # the second instance with the longest optimal solution 31
        [3,5,6,8,4,7,2,1,0],
        # [8,7,6,0,4,1,2,5,3], # the first instance with the longest optimal solution 31
        [1,8,6,7,4,3,2,5,0],
        # [8,5,6,7,2,3,4,1,0], # the first instance with the most solutions
        [8,7,4,5,6,1,2,3,0],
        # [8,5,4,7,6,3,2,1,0], # the second instance with the most solutions
        [8,7,6,5,2,1,4,3,0],
        # [8,6,7,2,5,4,3,0,1], # the "wrong"? hardest eight-puzzle from
        [7,8,3,6,5,4,1,2,0],
        # [6,4,7,8,5,0,3,2,1], # w01fe.com/blog/2009/01/the-hardest-eight-puzzle-instances-take-31-moves-to-solve/
        [5,8,7,6,1,4,0,2,3],
    ]
    gcs = np.arange(width*height).reshape((1,width*height))
    generate(p, ics, gcs, width, height)

def hanoi(disks=5, towers=3):
    import latplan.puzzles.hanoi as p
    ics = [
        np.zeros(disks,dtype=int),
        *[
            random_walk(np.full(disks,towers-1,dtype=int), steps, lambda config: p.successors(config,disks,towers))
            for i in range(instances-1)
        ]
    ]
    gcs = np.full((1,disks),towers-1,dtype=int)
    generate(p, ics, gcs, disks, towers)

def lightsout(type='digital', size=4):
    import importlib
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    ics = [
        random_walk(np.full(size*size,-1), steps, lambda config: p.successors(config))
        for i in range(instances)
    ]
    gcs = np.full((1,size*size),-1)
    generate(p, ics, gcs)

################################################################

def noise(fn, param, domain, *args):
    noise_fns.append(lambda a: fn(a,param))
    output_dirs.append(fn.__name__)
    domain(*args)

################################################################

def main():
    import sys
    try:
        print('args:',sys.argv)
        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        global steps, instances
        steps = myeval(sys.argv[1])
        instances = myeval(sys.argv[2])
        task      = myeval(sys.argv[3])
    except:
        print(sys.argv[0], 'steps','instances','task','[task-specific args...]')
    task(*map(myeval,sys.argv[4:]))

if __name__ == '__main__':
    main()


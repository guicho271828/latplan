#!/usr/bin/env python3

import numpy as np
import os
import os.path
import sys
sys.path.append('../../')

steps     = 5
instances = 100

def random_walk(init_c,length,successor_fn):
    print(".",end="")
    return random_walk_rec(init_c, [init_c], length, successor_fn)

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
            if np.any([np.all(np.equal(suc, t)) for t in trace]):
                now = (now+1)%len(sucs)
                if now == first:
                    print("B",end="")
                    return None
                else:
                    continue
            result = random_walk_rec(suc, [*trace, suc], length-1, successor_fn)
            if result is not None:
                return result
            else:
                continue

def generate(p, ics, gcs, *args):
    from scipy import misc
    import subprocess
    subprocess.call(["rm","-rf",p.__name__])
    for i,init in enumerate(p.generate(np.array(ics),*args)):
        for j,goal in enumerate(p.generate(np.array(gcs),*args)):
            d = "{}/{:03d}-{:03d}-{:03d}".format(p.__name__,steps,i,j)
            os.makedirs(d)
            print(d)
            misc.imsave(os.path.join(d,"init.png"),init)
            misc.imsave(os.path.join(d,"goal.png"),goal)

################################################################

def puzzle(type='mnist', width=3, height=3):
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def predefined_ics():
        return [
            # from Reinfield '93
            [8,0,6,5,4,7,2,3,1], # the second instance with the longest optimal solution 31
            [8,7,6,0,4,1,2,5,3], # the first instance with the longest optimal solution 31
            [8,5,6,7,2,3,4,1,0], # the first instance with the most solutions
            [8,5,4,7,6,3,2,1,0], # the second instance with the most solutions
            [8,6,7,2,5,4,3,0,1], # the "wrong"? hardest eight-puzzle from
            [6,4,7,8,5,0,3,2,1], # w01fe.com/blog/2009/01/the-hardest-eight-puzzle-instances-take-31-moves-to-solve/
        ]

    ics = [
        random_walk(np.arange(width*height), steps, lambda config: p.successors(config,width,height))
        for i in range(instances)
    ]
    gcs = np.arange(width*height).reshape((1,width*height))
    generate(p, ics, gcs, width, height)

def puzzle_longest(type='mnist', width=3, height=3):
    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    ics = [
        # from Reinfield '93
        [8,0,6,5,4,7,2,3,1], # the second instance with the longest optimal solution 31
        [8,7,6,0,4,1,2,5,3], # the first instance with the longest optimal solution 31
        [8,5,6,7,2,3,4,1,0], # the first instance with the most solutions
        [8,5,4,7,6,3,2,1,0], # the second instance with the most solutions
        [8,6,7,2,5,4,3,0,1], # the "wrong"? hardest eight-puzzle from
        [6,4,7,8,5,0,3,2,1], # w01fe.com/blog/2009/01/the-hardest-eight-puzzle-instances-take-31-moves-to-solve/
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

def lightsout_digital(size=4):
    import latplan.puzzles.lightsout_digital as p
    ics = [
        random_walk(np.full(size*size,-1), steps, lambda config: p.successors(config))
        for i in range(instances)
    ]
    gcs = np.full((1,size*size),-1)
    generate(p, ics, gcs)

def lightsout_twisted(size=4):
    import latplan.puzzles.lightsout_twisted as p
    ics = [
        random_walk(np.full(size*size,-1), steps, lambda config: p.successors(config))
        for i in range(instances)
    ]
    gcs = np.full((1,size*size),-1)
    generate(p, ics, gcs)

def main():
    import sys
    try:
        print(os.path.split(__file__)[0])
        os.chdir(os.path.split(__file__)[0])
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


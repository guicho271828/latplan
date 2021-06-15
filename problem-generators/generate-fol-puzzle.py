#!/usr/bin/env python3

import random
import numpy as np
import imageio
import os
import os.path
import sys
import subprocess
import datetime
import importlib
sys.path.append('../../')
from latplan.util import curry
from latplan.util.noise import gaussian, salt, pepper, saltpepper
from latplan.util.search import dijkstra, reservoir_sampling, untuple
from latplan.puzzles.objutil import tiled_bboxes, image_to_tiled_objects, bboxes_to_coord

def noise(fn, param, domain, *args):
    noise_fns.append(lambda a: fn(a,param))
    output_dirs.append(fn.__name__)
    domain(*args)

def identity(x):
    return x

steps     = 5
instances = 100
noise_fns     = [identity]
output_dirs = ["vanilla"]


def generate(name, init_images, goal_images, inits, goals, **kwargs):
    for noise_fn,output_dir in zip(noise_fns,output_dirs):
        init_images = noise_fn(init_images) 
        goal_images = noise_fn(goal_images)
        inits = noise_fn(inits) 
        goals = noise_fn(goals)
        for i,(init_image,goal_image,init,goal) in enumerate(zip(init_images,goal_images,inits,goals)):
            d = "{}/{}/{:03d}-{:03d}".format(output_dir,name,steps,i)
            if os.path.isdir(d):
                subprocess.call(["mv",d,d+"_old_"+datetime.datetime.today().isoformat()])
            os.makedirs(d)
            print(d)
            imageio.imsave(os.path.join(d,"init.png"),init)
            imageio.imsave(os.path.join(d,"goal.png"),goal)
            np.savez_compressed(os.path.join(d,"objs.npz"),init=init,goal=goal,**kwargs)


################################################################

def puzzle(type='mnist', width=3, height=3):
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config,width,height)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.arange(width*height)
        return tuple(config)

    goal = goal_state()
    init_candidates = untuple(dijkstra(goal, steps, successor_fn))
    ics = reservoir_sampling(init_candidates, instances)
    gcs = [ goal for i in range(len(ics))] # note: not range(instances), because ics may be shorter depending on search depth
    init_images = np.expand_dims(p.generate(np.array(ics), width, height),-1)
    goal_images = np.expand_dims(p.generate(np.array(gcs), width, height),-1)
    inits = image_to_tiled_objects(init_images, p.setting['base'])
    goals = image_to_tiled_objects(goal_images, p.setting['base'])
    B, H, W, C = init_images.shape
    assert C == 1
    picsize = [H,W]
    bboxes = tiled_bboxes(B, height, width, p.setting['base'])
    coord  = bboxes_to_coord(bboxes)
    inits = np.concatenate([inits,coord], axis=-1)
    goals = np.concatenate([goals,coord], axis=-1)
    generate("-".join(["puzzle",type,str(width),str(height)]), init_images, goal_images, inits, goals, picsize=picsize)

def puzzle_random_goal(type='mnist', width=3, height=3):
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    def successor_fn(config):
        r = p.successors(config,width,height)
        return [tuple(e) for e in r]

    def goal_state():
        config = np.arange(width*height)
        np.random.shuffle(config)
        return tuple(config)

    gcs = [ goal_state() for i in range(instances)]
    ics = [ reservoir_sampling(untuple(dijkstra(goal, steps, successor_fn)), 1)[0] for goal in gcs ]
    init_images = np.expand_dims(p.generate(np.array(ics), width, height),-1)
    goal_images = np.expand_dims(p.generate(np.array(gcs), width, height),-1)
    inits = image_to_tiled_objects(init_images, p.setting['base'])
    goals = image_to_tiled_objects(goal_images, p.setting['base'])
    B, H, W, C = init_images.shape
    assert C == 1
    picsize = [H,W]
    bboxes = tiled_bboxes(B, height, width, p.setting['base'])
    coord  = bboxes_to_coord(bboxes)
    inits = np.concatenate([inits,coord], axis=-1)
    goals = np.concatenate([goals,coord], axis=-1)
    generate("-".join(["puzzle_random_goal",type,str(width),str(height)]), init_images, goal_images, inits, goals, picsize=picsize)


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
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()


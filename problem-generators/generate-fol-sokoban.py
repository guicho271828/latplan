#!/usr/bin/env python3


import os
import numpy as np
import imageio
import gym
import pddlgym
from latplan.puzzles.objutil import tiled_bboxes, image_to_tiled_objects, bboxes_to_coord

from latplan.puzzles.sokoban import make_env, shrink, compute_relevant, tile, plan_to_actions

def sokoban(i,test):
    env, _ = make_env(i, test)

    init_state, debug_info = env.reset()
    init_image = shrink(env.render(mode="human_crisp"))
    init_layout = env.render(mode="layout")

    relevant = compute_relevant(init_layout)

    plan = pddlgym.planning.run_planner(debug_info['domain_file'], debug_info['problem_file'], "ff")
    actions = plan_to_actions(env,init_state,plan)

    for action in actions:
        _, _, done, _ = env.step(action)
        print(action,done)

    goal_image = shrink(env.render(mode="human_crisp"))

    images = np.stack([init_image, goal_image],axis=0)
    images = images / 255       # this is done in main.py before training

    B, H, W, C = images.shape

    picsize= [H,W,C]
    images = image_to_tiled_objects(images, tile)
    bboxes = tiled_bboxes(B, H//tile, W//tile, tile)
    coord  = bboxes_to_coord(bboxes)
    inputs = np.concatenate([images,coord],axis=-1)
    print(images.shape,bboxes.shape)

    inputs = inputs[:,relevant]
    print(inputs.shape)

    if test:
        d = f"sokoban-test/{i}/"
    else:
        d = f"sokoban/{i}/"
    if not os.path.isdir(d):
        os.makedirs(d)
    np.savez_compressed(os.path.join(d,"objs.npz"),init=inputs[0],goal=inputs[1],picsize=picsize)
    imageio.imwrite(os.path.join(d,"init.png"), init_image)
    imageio.imwrite(os.path.join(d,"goal.png"), goal_image)
    return

for i in range(5):
    sokoban(i,False)

for i in range(4):
    sokoban(i,True)


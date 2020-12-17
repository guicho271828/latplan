#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as nr
import random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import grid_search, nn_task
from latplan.util.noise  import gaussian
from latplan.util.search import dijkstra
from latplan.puzzles.objutil import tiled_bboxes, image_to_tiled_objects, bboxes_to_coord

import keras.backend as K
import tensorflow as tf

from skimage.transform import resize

import os
import os.path

import tqdm

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



################################################################
# sokoban: use logic in pddlgym

def render_sokoban(inputs):
    pairs,image_mode = inputs
    import gym
    import pddlgym
    pre_images  = []
    suc_images  = []
    env = gym.make("PDDLEnvSokoban-v0")
    env.reset()

    def shrink(x):
        x = x[:,:,:3]
        x = x * 256
        x = x.astype("uint8")
        return x

    for pobs, obs in pairs:
        env.set_state(pobs)
        pre_images.append(shrink(env.render(mode=image_mode)))

        env.set_state(obs)
        suc_images.append(shrink(env.render(mode=image_mode)))

    return pre_images, suc_images

def compute_reachability_sokoban(wall,player):
    h, w = wall.shape

    wall2     = np.zeros((h+2,w+2),dtype=bool)
    reachable = np.zeros((h+2,w+2),dtype=bool)
    wall2[1:h+1,1:w+1] = wall
    reachable[1:h+1,1:w+1] = player

    changed = True
    while changed:
        changed = False
        for y in range(h):
            for x in range(w):
                if not wall2[y+1,x+1] and \
                   ( reachable[y+1,x+2] or \
                     reachable[y+1,x  ] or \
                     reachable[y+2,x+1] or \
                     reachable[y,  x+1] ) and \
                     not reachable[y+1,x+1]:
                    reachable[y+1,x+1] = True
                    changed = True
    return reachable[1:h+1,1:w+1]


# stores images in an archive
def sokoban_image(limit = 1000, egocentric = False, objects = True, stage=0, test=False):
    list = ["sokoban_image",limit,
            ("egocentric" if egocentric else "global"),
            ("object"     if objects    else "global"),
            stage,
            ("test" if test else "train"),]
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,list))+".npz")
    import gym
    import pddlgym
    import imageio
    pre_images     = []
    suc_images     = []
    if egocentric:
        image_mode  = "egocentric_crisp"
    else:
        image_mode  = "human_crisp"

    env = gym.make("PDDLEnvSokoban-v0" if not test else "PDDLEnvSokobanTest-v0")
    env.fix_problem_index(stage)
    init, _ = env.reset()
    init_layout = env.render(mode="layout")

    # reachability analysis
    player = (init_layout == pddlgym.rendering.sokoban.PLAYER)
    wall   = (init_layout == pddlgym.rendering.sokoban.WALL)
    reachable = compute_reachability_sokoban(wall,player)
    relevant = np.maximum(reachable, wall)
    print(f"{wall.sum()} wall objects:")
    print(wall)
    print(f"{reachable.sum()} reachable objects:")
    print(reachable)
    print(f"{relevant.sum()} relevant objects:")
    print(relevant)
    relevant = relevant.reshape(-1)

    def successor(obs):
        env.set_state(obs)
        for action in env.action_space.all_ground_literals(obs, valid_only=True):
            env.set_state(obs)
            obs2, _, _, _ = env.step(action)
            yield obs2

    pairs = []
    max_g = 0
    for obs, close_list in dijkstra(init, float("inf"), successor, include_nonleaf=True, limit=limit):
        max_g = max(max_g,close_list[obs]["g"])
        pobs = close_list[obs]["parent"]
        if pobs is None:
            continue
        pairs.append((pobs,obs))

    threads = 16
    pairss = []
    len_per_thread = 1+(len(pairs) // threads)
    for i in range(threads):
        pairss.append(pairs[i*len_per_thread:(i+1)*len_per_thread])

    from multiprocessing import Pool
    with Pool(threads) as p:
        for sub in tqdm.tqdm(p.imap(render_sokoban,
                                    zip(pairss,
                                        [image_mode]*threads))):
            pre_images_sub  = sub[0]
            suc_images_sub  = sub[1]
            pre_images.extend(pre_images_sub)
            suc_images.extend(suc_images_sub)

    pre_images = np.array(pre_images)
    suc_images = np.array(suc_images)
    print(pre_images.shape)
    print("max",pre_images.max(),"min",pre_images.min())

    # shuffling
    random_indices = np.arange(len(pre_images))
    nr.shuffle(random_indices)
    pre_images = pre_images[random_indices]
    suc_images = suc_images[random_indices]

    if not objects:
        # whole image
        np.savez_compressed(path,pres=pre_images,sucs=suc_images)
        return

    # image
    tile = 16
    B, H, W, C = pre_images.shape

    pre_images = image_to_tiled_objects(pre_images, tile)
    suc_images = image_to_tiled_objects(suc_images, tile)
    bboxes = tiled_bboxes(B, H//tile, W//tile, tile)
    print(pre_images.shape,bboxes.shape)

    # prune the unreachable regions
    if not egocentric:
        pre_images = pre_images[:,relevant]
        suc_images = suc_images[:,relevant]
        bboxes = bboxes[:,relevant]
        print(pre_images.shape,bboxes.shape)

    # note: bbox can be reused for pres and sucs
    picsize = [H,W,C]
    np.savez_compressed(path,pres=pre_images,sucs=suc_images,bboxes=bboxes,picsize=picsize,max_g=max_g)


# stores state layouts in an archive.
# each state is represented as an array (H,W,num_classes) (one-hot in the last dimension).
def sokoban_layout(limit = 1000, egocentric = False, objects = True, stage=0, test=False):
    assert objects
    list = ["sokoban_layout",limit,
            ("egocentric" if egocentric else "global"),
            ("object"     if objects    else "global"),
            stage,
            ("test" if test else "train"),]
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,list))+".npz")
    import gym
    import pddlgym
    import imageio
    pre_layouts     = []
    suc_layouts     = []
    if egocentric:
        layout_mode = "egocentric_layout"
    else:
        layout_mode = "layout"

    env = gym.make("PDDLEnvSokoban-v0" if not test else "PDDLEnvSokobanTest-v0")
    env.fix_problem_index(stage)
    init, _ = env.reset()
    init_layout = env.render(mode=layout_mode)

    # reachability analysis
    player = (init_layout == pddlgym.rendering.sokoban.PLAYER)
    wall   = (init_layout == pddlgym.rendering.sokoban.WALL)
    reachable = compute_reachability_sokoban(wall,player)
    relevant = np.maximum(reachable, wall)
    print(f"{wall.sum()} wall objects:")
    print(wall)
    print(f"{reachable.sum()} reachable objects:")
    print(reachable)
    print(f"{relevant.sum()} relevant objects:")
    print(relevant)
    relevant = relevant.reshape(-1)

    def successor(obs):
        env.set_state(obs)
        for action in env.action_space.all_ground_literals(obs, valid_only=True):
            env.set_state(obs)
            obs2, _, _, _ = env.step(action)
            yield obs2

    max_g = 0
    for obs, close_list in dijkstra(init, float("inf"), successor, include_nonleaf=True, limit=limit):
        max_g = max(max_g,close_list[obs]["g"])
        pobs = close_list[obs]["parent"]
        if pobs is None:
            continue
        env.set_state(pobs)
        pre_layouts.append(env.render(mode=layout_mode))

        env.set_state(obs)
        suc_layouts.append(env.render(mode=layout_mode))

    pre_layouts = np.array(pre_layouts)
    suc_layouts = np.array(suc_layouts)
    print(pre_layouts.shape)
    print("max",pre_layouts.max(),"min",pre_layouts.min())
    B, H, W = pre_layouts.shape
    pre_layouts = pre_layouts.reshape((B,H*W))
    suc_layouts = suc_layouts.reshape((B,H*W))

    # shuffling
    random_indices = np.arange(len(pre_layouts))
    nr.shuffle(random_indices)
    pre_layouts = pre_layouts[random_indices]
    suc_layouts = suc_layouts[random_indices]

    tile = 16
    bboxes = tiled_bboxes(B, H, W, tile)

    if not egocentric:
        pre_layouts = pre_layouts[:,relevant]
        suc_layouts = suc_layouts[:,relevant]
        bboxes = bboxes[:,relevant]

    # make it into a one-hot repr
    eye = np.eye(pddlgym.rendering.sokoban.NUM_OBJECTS)
    # B, H, W, C
    pre_classes = eye[pre_layouts]
    suc_classes = eye[suc_layouts]
    print(pre_classes.shape)

    np.savez_compressed(path,pres=pre_classes,sucs=suc_classes,bbox=bboxes,picsize=[H*tile,W*tile,3],max_g=max_g)


################################################################

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

#!/usr/bin/env python3

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

import os
import os.path

import tqdm

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

inf = float("inf")

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

from latplan.puzzles.sokoban import archive_path, make_env, shrink, compute_relevant, tile

# to perform rendering in multiprocess, the function must be global
def render_sokoban(inputs):
    pairs,image_mode = inputs
    import gym
    import pddlgym
    pre_images  = []
    suc_images  = []
    env = gym.make("PDDLEnvSokoban-v0")
    env.reset()

    for pobs, obs in pairs:
        env.set_state(pobs)
        pre_images.append(shrink(env.render(mode=image_mode)))

        env.set_state(obs)
        suc_images.append(shrink(env.render(mode=image_mode)))

    return pre_images, suc_images


# stores images in an archive
def sokoban_image(limit = 1000, egocentric = False, objects = True, stage=0, test=False):
    path = archive_path("image",limit,egocentric,objects,stage,test)
    pre_images     = []
    suc_images     = []
    if egocentric:
        image_mode  = "egocentric_crisp"
    else:
        image_mode  = "human_crisp"

    env, successor = make_env(stage, test)
    init, _ = env.reset()
    init_layout = env.render(mode="layout")

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

    # image
    B, H, W, C = pre_images.shape
    picsize = [H,W,C]

    if not objects:
        # whole image
        pre_images = pre_images.reshape((B,1,-1))
        suc_images = suc_images.reshape((B,1,-1))
        bboxes  = np.zeros((B, 1, 4))
        bboxes[:,0] = [0,0,H,W]
        np.savez_compressed(path,pres=pre_images,sucs=suc_images,bboxes=bboxes,picsize=picsize,max_g=max_g)
        return

    pre_images = image_to_tiled_objects(pre_images, tile)
    suc_images = image_to_tiled_objects(suc_images, tile)
    bboxes = tiled_bboxes(B, H//tile, W//tile, tile)
    print(pre_images.shape,bboxes.shape)

    if not egocentric:
        # prune unreachable regions
        relevant = compute_relevant(init_layout)
        pre_images = pre_images[:,relevant]
        suc_images = suc_images[:,relevant]
        bboxes = bboxes[:,relevant]
        print(pre_images.shape,bboxes.shape)

    # note: bbox can be reused for pres and sucs
    np.savez_compressed(path,pres=pre_images,sucs=suc_images,bboxes=bboxes,picsize=picsize,max_g=max_g)
    return


# stores state layouts in an archive.
# each state is represented as an array (H,W,1), where each data is an integer 0 <= x < pddlgym.rendering.sokoban.NUM_OBJECTS .
# i.e., the data is treated as a single channel image.
def sokoban_layout(limit = 1000, egocentric = False, objects = True, stage=0, test=False):
    path = archive_path("layout",limit,egocentric,objects,stage,test)
    pre_layouts     = []
    suc_layouts     = []
    if egocentric:
        layout_mode = "egocentric_layout"
    else:
        layout_mode = "layout"

    env, successor = make_env(stage, test)
    init, _ = env.reset()
    init_layout = env.render(mode=layout_mode)

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

    pre_layouts = np.array(pre_layouts,dtype=np.int8)
    suc_layouts = np.array(suc_layouts,dtype=np.int8)
    print(pre_layouts.shape)
    print("max",pre_layouts.max(),"min",pre_layouts.min())

    # shuffling
    random_indices = np.arange(len(pre_layouts))
    nr.shuffle(random_indices)
    pre_layouts = pre_layouts[random_indices]
    suc_layouts = suc_layouts[random_indices]

    B, H, W = pre_layouts.shape
    picsize = [H,W,1]

    if not objects:
        # whole layout
        pre_layouts = pre_layouts.reshape((B,1,-1))
        suc_layouts = suc_layouts.reshape((B,1,-1))
        bboxes  = np.zeros((B, 1, 4))
        bboxes[:,0] = [0,0,H,W]
        np.savez_compressed(path,pres=pre_layouts,sucs=suc_layouts,bboxes=bboxes,picsize=picsize,max_g=max_g)
        return

    pre_layouts = pre_layouts.reshape((B,H*W,1))
    suc_layouts = suc_layouts.reshape((B,H*W,1))
    bboxes = tiled_bboxes(B, H, W, tile)

    if not egocentric:
        # prune unreachable regions
        relevant = compute_relevant(init_layout)
        pre_layouts = pre_layouts[:,relevant]
        suc_layouts = suc_layouts[:,relevant]
        bboxes = bboxes[:,relevant]
        print(pre_layouts.shape,bboxes.shape)

    np.savez_compressed(path,pres=pre_layouts,sucs=suc_layouts,bbox=bboxes,picsize=picsize,max_g=max_g)


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

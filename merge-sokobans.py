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
from latplan.util.stacktrace import format
from latplan.puzzles.objutil import tiled_bboxes, image_to_tiled_objects, bboxes_to_coord, random_object_masking, location_augmentation

import keras.backend as K
import tensorflow as tf

from skimage.transform import resize

import os
import os.path

from tqdm import tqdm

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(threshold=sys.maxsize,formatter={'float_kind':float_formatter})

################################################################

# no egocentric, no global
def sokoban_image(limit = 1000, egocentric = False, objects=True, test=False):
    import gym
    import pddlgym

    all_transitions = []
    all_picsizes = []
    for stage in tqdm(range(5)):
        list = ["sokoban_image",limit,
                ("egocentric" if egocentric else "global"),
                ("object"     if objects    else "global"),
                stage,
                ("test" if test else "train"),]
        path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,list))+".npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as data:
            pres = data['pres'] # [B,25,sH*sW*C]
            sucs = data['sucs'] # [B,25,sH*sW*C]
            bboxes = data['bboxes'] # [B,25,4]
            pres = np.concatenate([pres,bboxes],axis=-1) # B,O,F
            sucs = np.concatenate([sucs,bboxes],axis=-1) # B,O,F
            transitions = np.stack([pres,sucs],axis=1)   # B,2,O,F
            all_transitions.append(transitions)
            all_picsizes.append(data['picsize'])
            print(f"loaded {path}")

    print(all_picsizes)
    new_picsize = np.max(np.stack(all_picsizes),axis=0) # [5,3] -> [3]
    min_objects = np.min([ x.shape[2] for x in all_transitions ])
    min_size    = np.min([ x.shape[0] for x in all_transitions ])
    max_objects = np.max([ x.shape[2] for x in all_transitions ])
    max_size    = np.max([ x.shape[0] for x in all_transitions ])
    print("min_objects:",min_objects)
    print("max_objects:",max_objects)
    print("min_size:",min_size)
    print("max_size:",max_size)
    print("new_picsize:",new_picsize)

    all_masked_transitions=[]
    for i, picsize, transitions in zip(range(5), all_picsizes, all_transitions):
        print(i,"0:", transitions.shape)
        # avoid imbalance
        ids = np.arange(len(transitions))
        nr.shuffle(ids)
        transitions = transitions[ids[:min_size]]

        print(i,"1:", transitions.shape)
        # standardize the number of objects
        if transitions.shape[2] != min_objects:
            transitions = random_object_masking(transitions,min_objects)

        print(i,"2:", transitions.shape)

        # move the global coordinate to the center
        picsize_diff = new_picsize - picsize
        dH, dW, _ = picsize_diff
        transitions[...,-4] += dW//2
        transitions[...,-3] += dH//2
        transitions[...,-2] += dW//2
        transitions[...,-1] += dH//2
        
        # move the global coordinate of the environment randomly
        # in order to evenly cover the maximum canvas size
        # picsize_diff = new_picsize - picsize
        # transitions = location_augmentation(transitions,
        #                                     height=picsize_diff[0],
        #                                     width=picsize_diff[1])
        print(i,"3:", transitions.shape)
        all_masked_transitions.append(transitions)


    masked_transitions = np.concatenate(all_masked_transitions)
    shape = masked_transitions.shape

    # shuffle so that different problem instances appear in a round-robin manner
    masked_transitions = np.reshape(masked_transitions,
                                    (len(all_masked_transitions), # problem
                                     min_size,                    # size
                                     *shape[1:]))
    masked_transitions = np.swapaxes(masked_transitions, 0, 1)
    masked_transitions = np.reshape(masked_transitions,
                                    (-1,*shape[1:]))

    masked_pres       = masked_transitions[:,0]
    masked_sucs       = masked_transitions[:,1]
    masked_pres_image = masked_pres[:,:,:-4]
    masked_sucs_image = masked_sucs[:,:,:-4]
    masked_bbox       = masked_sucs[:,:,-4:]
    print("masked_pres_image",masked_pres_image.shape)
    print("masked_sucs_image",masked_sucs_image.shape)
    print("masked_bbox      ",masked_bbox      .shape)
    list = ["sokoban_image",limit,
            "global",
            "object",
            "merged",
            ("test" if test else "train"),]
    np.savez_compressed(
        os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,list))+".npz"),
        pres    = masked_pres_image,
        sucs    = masked_sucs_image,
        bboxes  = masked_bbox,
        picsize = new_picsize)


def main():
    import sys
    if len(sys.argv) == 1:
        print({ k for k in dir(latplan.model)})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str

        sokoban_image(*map(myeval,sys.argv))


if __name__ == '__main__':
    try:
        main()
    except:
        format()

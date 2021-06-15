
import random
import numpy as np
from .model.hanoi import generate_configs, generate_random_configs, successors, config_state
from .model.util import binary_search
import math
from .util import wrap
from .util import preprocess

from keras.layers import Input
from keras.models import Model
import keras.backend.tensorflow_backend as K
import tensorflow as tf

## code ##############################################################

# see model/hanoi.py for the details of config encoding.
# this file contains only the rendering and validation code.

disk_height = 1                 # height of a single disk in pixel.
disk_inc = 0                    # the increment of the width between disks in pixel. The actual width increases twice this amount because it is added on both the left and the right sides.
base_disk_width_factor = 1      # ratio between the disk height and width. larger factor = more width
base_disk_width = 3 + disk_height * base_disk_width_factor # the width of the smallest disk.

colors = [
    # r,g,b
    # [0,0,0],
    [1,1,1],
    [1,0,0],
    [0,1,0],
    [0,0,1],
    [1,1,0],
    [0,1,1],
    [1,0,1],
    [1,0.5,0],
    [0.5,1,0],
    [0,1,0.5],
    [0,0.5,1],
    [0.5,0,1],
    [1,0,0.5],
]

setting = {
    'min_threshold' : 0.0,
    'max_threshold' : 0.5,
}

def generate1(config,disks,towers, **kwargs):
    l = len(config)
    tower_width  = disks * (2*disk_inc) + base_disk_width
    tower_height = disks*disk_height
    figure = np.full([tower_height, tower_width*towers, 3], 0.5) # gray
    state = config_state(config,disks,towers)
    for i, tower in enumerate(state):
        tower.reverse()
        x_center = tower_width *  i + disks * disk_inc # lacks base_disk_width
        for j,disk in enumerate(tower):
            # print(j,disk,(l-j)*2)
            figure[
                tower_height - disk_height * (j+1) :
                tower_height - disk_height * j,
                x_center - disk * disk_inc :
                x_center + disk * disk_inc + base_disk_width] \
                = colors[disk]
    return preprocess(figure)

def generate(configs,disks,towers, **kwargs):
    return np.array([ generate1(c,disks,towers, **kwargs) for c in configs ])
                

def states(disks, towers, configs=None, **kwargs):
    if configs is None:
        configs = generate_configs(disks, towers)
    return generate(configs,disks,towers, **kwargs)

def transitions_old(disks, towers, configs=None, one_per_state=False, **kwargs):
    if configs is None:
        configs = generate_configs(disks, towers)
    if one_per_state:
        transitions = np.array([
            generate([c1,random.choice(successors(c1,disks,towers))],disks,towers, **kwargs)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],disks,towers, **kwargs)
                                 for c1 in configs for c2 in successors(c1,disks,towers) ])
    return np.einsum('ab...->ba...',transitions)

def transitions(disks, towers, configs=None, one_per_state=False, **kwargs):
    if configs is None:
        configs = generate_configs(disks, towers)
    if one_per_state:
        pre = generate(configs, disks, towers, **kwargs)
        suc = generate(np.array([random.choice(successors(c1, disks, towers)) for c1 in configs ]), disks, towers, **kwargs)
        return np.array([pre, suc])
    else:
        transitions = np.array([ [c1,c2] for c1 in configs for c2 in successors(c1, disks, towers) ])
        pre = generate(transitions[:,0,:], disks, towers, **kwargs)
        suc = generate(transitions[:,1,:], disks, towers, **kwargs)
        return np.array([pre, suc])

def setup():
    pass

def get_panels(disks, tower_width):
    # build a single tower with all disks, plus one empty row
    panels = np.full([disks+1, disk_height, tower_width, 3], 0.5)
    x_center = disks * disk_inc # lacks base_disk_width
    for disk, panel in enumerate(panels):
        if disk != disks:       # except the last empty panel
            panel[:,
                x_center - disk * disk_inc :
                x_center + disk * disk_inc + base_disk_width] \
                = colors[disk]
    return panels


def build_error(s, disks, towers, tower_width, panels):
    # s: [batch,disks*height,towers*width,3]
    s = K.reshape(s,[-1,disks, disk_height, towers, tower_width, 3])
    s = K.permute_dimensions(s, [0,1,3,2,4,5])
    s = K.reshape(s,[-1,disks,towers, 1,      disk_height,tower_width,3])
    s = K.tile   (s,[1, 1,    1,      disks+1,1,          1,          1])
    # s: [batch,disks,towers,disks+1,height,width,3]

    # allpanels: [disks+1,height,width,3]
    allpanels = K.variable(panels)
    allpanels = K.reshape(allpanels, [1,1,1,disks+1,disk_height,tower_width,3])
    allpanels = K.tile(allpanels, [K.shape(s)[0], disks, towers, 1, 1, 1, 1])
    # allpanels: [batch,disks,towers,disks+1,height,width,3]

    error = K.abs(s - allpanels)
    error = K.mean(error, axis=(4,5,6))
    # error: [batch,disks,towers,disks+1]
    return error
    
def validate_states(states,verbose=True, **kwargs):
    tower_height = states.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width
    towers = states.shape[2] // tower_width
    panels = get_panels(disks, tower_width)

    bs = binary_search(setting["min_threshold"],setting["max_threshold"])
    def build():
        states = Input(shape=(tower_height, tower_width*towers, 3))
        error = build_error(states, disks, towers, tower_width, panels)
        matches = 1 - K.clip(K.sign(error - bs.value),0,1)

        num_matches = K.sum(matches, axis=3)
        panels_ok = K.all(K.equal(num_matches, 1), (1,2)) # there is exactly one matched panel
        panels_ng = K.any(K.not_equal(num_matches, 1), (1,2)) # number of matches are not 1
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2)) # there are no matched panel
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2)) # there are more than one matched panels

        panel_coverage = K.sum(matches,axis=(1,2))
        # ideally, this should be [[1,1,1...1,1,1,disks*tower-disk], ...]

        # there should be one match each + disks*towers-disks matches for the background
        ideal_coverage = np.ones(disks+1)
        ideal_coverage[-1] = disks*towers-disks
        ideal_coverage = K.variable(ideal_coverage)
        coverage_ok = K.all(K.equal(panel_coverage, ideal_coverage), 1)
        coverage_ng = K.any(K.not_equal(panel_coverage, ideal_coverage), 1)
        validity = tf.logical_and(panels_ok, coverage_ok)

        return Model(states,
                     [ wrap(states, x) for x in [panels_ok,
                                                 panels_nomatch,
                                                 panels_ambiguous,
                                                 coverage_ok,
                                                 coverage_ng,
                                                 validity]])

    while True:
        model = build()
        panels_ok, panels_nomatch, panels_ambiguous, \
            coverage_ok, coverage_ng, validity = model.predict(states, **kwargs)
        panels_nomatch = np.count_nonzero(panels_nomatch)
        panels_ambiguous = np.count_nonzero(panels_ambiguous)
        if verbose:
            print(f"threshold value: {bs.value}")
            print(panels_nomatch,                    "images have some panels which are unlike any panels")
            print(np.count_nonzero(panels_ok),       "images have all panels which match exactly 1 panel each")
            print(panels_ambiguous,                  "images have some panels which match >2 panels")
        if np.abs(panels_nomatch - panels_ambiguous) <= 1:
            if verbose:
                print("threshold found")
                print(np.count_nonzero(np.logical_and(panels_ok, coverage_ng)),"images have duplicated tiles")
                print(np.count_nonzero(np.logical_and(panels_ok, coverage_ok)),"images have no duplicated tiles")
            return validity
        elif panels_nomatch < panels_ambiguous:
            bs.goleft()
        else:
            bs.goright()


def to_configs(states,verbose=True, **kwargs):
    tower_height = states.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width
    towers = states.shape[2] // tower_width
    panels = get_panels(disks, tower_width)

    bs = binary_search(setting["min_threshold"],setting["max_threshold"])
    def build():
        states = Input(shape=(tower_height, tower_width*towers, 3))
        error = build_error(states, disks, towers, tower_width, panels)
        # error: [batch,disks,towers,disks+1]
        matches = 1 - K.clip(K.sign(error - bs.value),0,1)
        # matches: [batch,disks,towers,disks+1]

        num_matches = K.sum(matches, axis=3)
        panels_ok = K.all(K.equal(num_matches, 1), (1,2)) # there is exactly one matched panel
        panels_ng = K.any(K.not_equal(num_matches, 1), (1,2)) # number of matches are not 1
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2)) # there are no matched panel
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2)) # there are more than one matched panels

        panel_coverage = K.sum(matches,axis=(1,2))
        # ideally, this should be [[1,1,1...1,1,1,disks*tower-disk], ...]

        # there should be one match each + disks*towers-disks matches for the background
        ideal_coverage = np.ones(disks+1)
        ideal_coverage[-1] = disks*towers-disks
        ideal_coverage = K.variable(ideal_coverage)
        coverage_ok = K.all(K.equal(panel_coverage, ideal_coverage), 1)
        coverage_ng = K.any(K.not_equal(panel_coverage, ideal_coverage), 1)
        validity = tf.logical_and(panels_ok, coverage_ok)

        # assume disks=4, towers=3
        # matches: a h w p
        # [[[00001][00001][00001]]  --- all panel 4 (empty panel)
        #  [[10000][00001][00001]]  --- h,w=1,0 is panel 0, others are panel 4 (empty panel)
        #  [[01000][00001][00001]]  --- h,w=2,0 is panel 1, others are panel 4 (empty panel)
        #  [[00010][00100][00001]]] --- h,w=3,0 is panel 3, h,w=3,1 is panel 2, h,w=3,2 is panel 4
        # 
        # target config is [0,0,1,0]

        # you don't need the last panel (empty panel)
        # a h w p
        # [[[0000][0000][0000]]  --- all panel 4 (empty panel)
        #  [[1000][0000][0000]]  --- h,w=1,0 is panel 0, others are panel 4 (empty panel)
        #  [[0100][0000][0000]]  --- h,w=2,0 is panel 1, others are panel 4 (empty panel)
        #  [[0001][0010][0000]]] --- h,w=3,0 is panel 3, h,w=3,1 is panel 2, h,w=3,2 is panel 4
        config = matches[:, :, :, 0:-1]
        
        # you don't need the height info
        # a w p
        # [[1101][0010][0000]]
        config = K.sum(config, 1)

        # reorder to a p w
        # [[100][100][010][100]]
        config = K.permute_dimensions(config, [0,2,1])
        
        # convert one-hot width into width position
        config = config * K.arange(0,towers,dtype='float32') # 1-4
        # [[000][000][010][000]]
        config = K.sum(config, -1)
        # [0 0 1 0]
        config = K.cast(config, 'int32')

        return Model(states,
                     [ wrap(states, x) for x in [panels_ok,
                                                 panels_nomatch,
                                                 panels_ambiguous,
                                                 coverage_ok,
                                                 coverage_ng,
                                                 validity,
                                                 config]])

    while True:
        model = build()
        panels_ok, panels_nomatch, panels_ambiguous, \
            coverage_ok, coverage_ng, validity, config = model.predict(states, **kwargs)
        panels_nomatch = np.count_nonzero(panels_nomatch)
        panels_ambiguous = np.count_nonzero(panels_ambiguous)
        if verbose:
            print(f"threshold value: {bs.value}")
            print(panels_nomatch,                    "images have some panels which are unlike any panels")
            print(np.count_nonzero(panels_ok),       "images have all panels which match exactly 1 panel each")
            print(panels_ambiguous,                  "images have some panels which match >2 panels")
        if np.abs(panels_nomatch - panels_ambiguous) <= 1:
            if verbose:
                print("threshold found")
                print(np.count_nonzero(np.logical_and(panels_ok, coverage_ng)),"images have duplicated tiles")
                print(np.count_nonzero(np.logical_and(panels_ok, coverage_ok)),"images have no duplicated tiles")
            return config
        elif panels_nomatch < panels_ambiguous:
            bs.goleft()
        else:
            bs.goright()


def validate_transitions(transitions, check_states=True, **kwargs):
    pre = np.array(transitions[0])
    suc = np.array(transitions[1])

    tower_height = pre.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width
    towers = pre.shape[2] // tower_width
    
    if check_states:
        pre_validation = validate_states(pre, verbose=False, **kwargs)
        suc_validation = validate_states(suc, verbose=False, **kwargs)

    pre_configs = to_configs(pre, verbose=False, **kwargs)
    suc_configs = to_configs(suc, verbose=False, **kwargs)
    
    results = []
    if check_states:
        for pre_c, suc_c, pre_validation, suc_validation in zip(pre_configs, suc_configs, pre_validation, suc_validation):

            if pre_validation and suc_validation:
                succs = successors(pre_c, disks, towers)
                results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
            else:
                results.append(False)
    else:
        for pre_c, suc_c in zip(pre_configs, suc_configs):
            succs = successors(pre_c, disks, towers)
            results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
    return results



import numpy as np
from .model.hanoi import generate_configs, successors, config_state
import math
from .util import wrap
from .util import preprocess

from keras.layers import Input
from keras.models import Model
from keras import backend as K
import tensorflow as tf

## code ##############################################################

disk_height = 4
disk_inc = 2
base_disk_width_factor = 1
base_disk_width = disk_height * base_disk_width_factor

border = 0
tile_factor = 1

def generate1(config,disks,towers, **kwargs):
    l = len(config)
    tower_width  = disks * (2*disk_inc) + base_disk_width + border
    tower_height = disks*disk_height
    figure = np.ones([tower_height,
                      tower_width*towers],dtype=np.int8)
    state = config_state(config,disks,towers)
    for i, tower in enumerate(state):
        tower.reverse()
        # print(i,tower)
        x_center = tower_width *  i + disks * disk_inc # lacks base_disk_width
        for j,disk in enumerate(tower):
            # print(j,disk,(l-j)*2)
            figure[
                tower_height - disk_height * (j+1) :
                tower_height - disk_height * j,
                x_center - disk * disk_inc :
                x_center + disk * disk_inc + base_disk_width] \
                = 0
                # = np.tile(np.tile(patterns[disk],(tile_factor,tile_factor)),
                #           (1,2*disks+base_disk_width_factor))[:,:2 * disk * disk_inc + base_disk_width]
                # = np.tile(np.tile(patterns[disk],(tile_factor,tile_factor)),
                #           (1,disk+base_disk_width_factor))
                # = np.tile(np.tile(patterns[disk],(tile_factor,tile_factor)),
                #           (1,2*disk+base_disk_width_factor))
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
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate([c1,pickone(successors(c1,disks,towers))],disks,towers, **kwargs)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],disks,towers, **kwargs)
                                 for c1 in configs for c2 in successors(c1,disks,towers) ])
    return np.einsum('ab...->ba...',transitions)

def transitions(disks, towers, configs=None, one_per_state=False, **kwargs):
    if configs is None:
        configs = generate_configs(disks, towers)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        pre = generate(configs, disks, towers, **kwargs)
        suc = generate(np.array([pickone(successors(c1, disks, towers)) for c1 in configs ]), disks, towers, **kwargs)
        return np.array([pre, suc])
    else:
        transitions = np.array([ [c1,c2] for c1 in configs for c2 in successors(c1, disks, towers) ])
        pre = generate(transitions[:,0,:], disks, towers, **kwargs)
        suc = generate(transitions[:,1,:], disks, towers, **kwargs)
        return np.array([pre, suc])

def setup():
    pass

def get_panels(disks, tower_width):
    panels = np.ones([disks+1, disk_height, tower_width], dtype=np.int8)
    x_center = disks * disk_inc # lacks base_disk_width
    for disk, panel in enumerate(panels):
        if disk != disks:
            # last panel is empty
            panel[:,
                x_center - disk * disk_inc :
                x_center + disk * disk_inc + base_disk_width] \
                = 0
    return panels


threshold = 0.01
def build_error(s, disks, towers, tower_width, panels):
    s = K.reshape(s,[-1,disks, disk_height, towers, tower_width])
    s = K.permute_dimensions(s, [0,1,3,2,4])
    s = K.reshape(s,[-1,disks,towers,1,    disk_height,tower_width])
    s = K.tile   (s,[1, 1, 1, disks+1,1, 1,])

    allpanels = K.variable(panels)
    allpanels = K.reshape(allpanels, [1,1,1,disks+1,disk_height,tower_width])
    allpanels = K.tile(allpanels, [K.shape(s)[0], disks, towers, 1, 1, 1])

    def hash(x):
        ## 2x2 average hashing (now it does not work since disks have 1 pixel height)
        # x = K.reshape(x, [-1,disks,towers,disks+1, disk_height,tower_width//2,2])
        # x = K.mean(x, axis=(4,))
        # return K.round(x)
        ## diff hashing (horizontal diff)
        # x1 = x[:,:,:,:,:,:-1]
        # x2 = x[:,:,:,:,:,1:]
        # d = x1 - x2
        # return K.round(d)
        ## just rounding
        return K.round(x)
        ## do nothing
        # return x

    s         = hash(s)
    allpanels = hash(allpanels)
    
    # error = K.binary_crossentropy(s, allpanels)
    error = K.abs(s - allpanels)
    error = K.mean(error, axis=(4,5))
    return error
    
def validate_states(states,verbose=True, **kwargs):
    tower_height = states.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width + border
    towers = states.shape[2] // tower_width
    panels = get_panels(disks, tower_width)

    def build():
        states = Input(shape=(tower_height, tower_width*towers))
        error = build_error(states, disks, towers, tower_width, panels)
        matches = 1 - K.clip(K.sign(error - threshold),0,1)

        num_matches = K.sum(matches, axis=3)
        panels_ok = K.all(K.equal(num_matches, 1), (1,2))
        panels_ng = K.any(K.not_equal(num_matches, 1), (1,2))
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2))
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2))

        panel_coverage = K.sum(matches,axis=(1,2))
        # ideally, this should be [[1,1,1...1,1,1,disks*tower-disk], ...]
        
        ideal_coverage = np.ones(disks+1)
        ideal_coverage[-1] = disks*towers-disks
        ideal_coverage = K.variable(ideal_coverage)
        coverage_ok = K.all(K.equal(panel_coverage, ideal_coverage), 1)
        coverage_ng = K.any(K.not_equal(panel_coverage, ideal_coverage), 1)
        validity = tf.logical_and(panels_ok, coverage_ok)

        if verbose:
            return Model(states,
                         [ wrap(states, x) for x in [panels_ok,
                                                     panels_ng,
                                                     panels_nomatch,
                                                     panels_ambiguous,
                                                     coverage_ok,
                                                     coverage_ng,
                                                     validity]])
        else:
            return Model(states, wrap(states, validity))

    model = build()
    #     model.summary()
    if verbose:
        panels_ok, panels_ng, panels_nomatch, panels_ambiguous, \
            coverage_ok, coverage_ng, validity = model.predict(states, **kwargs)
        print(np.count_nonzero(panels_ng),       "images have some panels which match 0 or >2 panels, out of which")
        print(np.count_nonzero(panels_nomatch),  "images have some panels which are unlike any panels")
        print(np.count_nonzero(panels_ambiguous),"images have some panels which match >2 panels")
        print(np.count_nonzero(panels_ok),       "images have panels (all of them) which match exactly 1 panel each")
        print(np.count_nonzero(np.logical_and(panels_ok, coverage_ng)),"images have duplicated tiles")
        print(np.count_nonzero(np.logical_and(panels_ok, coverage_ok)),"images have no duplicated tiles")
        return validity
    else:
        validity = model.predict(states, **kwargs)
        return validity


def to_configs(states,verbose=True, **kwargs):
    tower_height = states.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width + border
    towers = states.shape[2] // tower_width
    panels = get_panels(disks, tower_width)

    def build():
        states = Input(shape=(tower_height, tower_width*towers))
        error = build_error(states, disks, towers, tower_width, panels)
        matches = 1 - K.clip(K.sign(error - threshold),0,1)
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
        return Model(states, wrap(states, config))

    return build().predict(states, **kwargs)

def validate_transitions(transitions, check_states=True, **kwargs):
    pre = np.array(transitions[0])
    suc = np.array(transitions[1])

    tower_height = pre.shape[1]
    disks = tower_height // disk_height
    
    tower_width  = disks * (2*disk_inc) + base_disk_width + border
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

## patterns ##############################################################

patterns = [
    [[0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],],
    [[1,1,1,1,1,1,1,1,],
     [1,1,1,0,0,1,1,1,],
     [1,1,1,0,0,1,1,1,],
     [1,1,1,0,0,1,1,1,],
     [1,1,1,0,0,1,1,1,],
     [1,0,0,0,0,0,0,1,],
     [1,0,0,0,0,0,0,1,],
     [1,1,1,1,1,1,1,1,],],
    [[1,1,0,0,1,1,0,0,],
     [1,1,0,0,1,1,0,0,],
     [0,0,1,1,0,0,1,1,],
     [0,0,1,1,0,0,1,1,],
     [1,1,0,0,1,1,0,0,],
     [1,1,0,0,1,1,0,0,],
     [0,0,1,1,0,0,1,1,],
     [0,0,1,1,0,0,1,1,],],
    [[1,1,1,1,1,1,1,1,],
     [1,1,0,0,0,0,1,1,],
     [1,0,1,1,1,1,0,1,],
     [1,0,1,1,1,1,0,1,],
     [1,0,1,1,1,1,0,1,],
     [1,0,1,1,1,1,0,1,],
     [1,1,0,0,0,0,1,1,],
     [1,1,1,1,1,1,1,1,],],
    [[0,0,0,1,0,0,0,1,],
     [0,0,1,0,0,0,1,0,],
     [0,1,0,0,0,1,0,0,],
     [1,0,0,0,1,0,0,0,],
     [0,0,0,1,0,0,0,1,],
     [0,0,1,0,0,0,1,0,],
     [0,1,0,0,0,1,0,0,],
     [1,0,0,0,1,0,0,0,],],
    [[1,1,1,1,1,1,1,1,],
     [1,1,0,0,0,0,1,1,],
     [1,0,0,0,0,0,0,1,],
     [1,0,0,0,0,0,0,1,],
     [1,0,0,0,0,0,0,1,],
     [1,0,0,0,0,0,0,1,],
     [1,1,0,0,0,0,1,1,],
     [1,1,1,1,1,1,1,1,],],
    [[0,0,0,0,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,1,1,1,0,0,0,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,1,1,1,0,],
     [0,0,0,0,0,0,0,0,],],
    [[1,0,0,0,1,0,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,0,1,0,0,0,1,],
     [1,0,0,0,1,0,0,0,],
     [0,1,0,0,0,1,0,0,],
     [0,0,1,0,0,0,1,0,],
     [0,0,0,1,0,0,0,1,],],
    [[0,0,0,1,1,0,0,0,],
     [0,0,1,0,0,1,0,0,],
     [0,1,0,0,0,0,1,0,],
     [1,0,0,0,0,0,0,1,],
     [0,0,0,1,1,0,0,0,],
     [0,0,1,0,0,1,0,0,],
     [0,1,0,0,0,0,1,0,],
     [1,0,0,0,0,0,0,1,],],
    [[1,0,0,0,0,0,0,1,],
     [0,1,0,0,0,0,1,0,],
     [0,0,1,0,0,1,0,0,],
     [0,0,0,1,1,0,0,0,],
     [1,0,0,0,0,0,0,1,],
     [0,1,0,0,0,0,1,0,],
     [0,0,1,0,0,1,0,0,],
     [0,0,0,1,1,0,0,0,],],
    [[1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],
     [1,0,1,0,1,0,1,0,],],
    [[1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],],
    [[0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [1,1,1,1,1,1,1,1,],
     [0,0,0,0,0,0,0,0,],
     [0,0,0,0,0,0,0,0,],],
    [[0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],
     [0,0,1,1,1,1,0,0,],],
]

patterns = np.array(patterns)

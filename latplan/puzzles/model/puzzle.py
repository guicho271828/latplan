#!/usr/bin/env python3

import random
import numpy as np
from ...util.np_distances import bce
from ..util import wrap

from keras.layers import Input, Reshape
from keras.models import Model
import keras.backend.tensorflow_backend as K
import tensorflow as tf

# domain specific state representation:
#
# In a config array C, C_ij is the location of j-th panel in the i-th configuration.
# 
# [[0,1,2,3,4,5,6,7,8]] represents a single configuration, where 0 is at 0 (top left)),
# 1 is at 1 (top middle) and so on.

setting = {
    'base' : None,
    'panels' : None,
    'loader' : None,
    'min_threshold' : 0.0,
    'max_threshold' : 0.5,
}

def load(width,height,force=False):
    if setting['panels'] is None or force is True:
        setting['panels'] = setting['loader'](width,height)

def generate(configs, width, height, **kwargs):
    load(width, height)

    from keras.layers import Input, Reshape
    from keras.models import Model
    import keras.backend.tensorflow_backend as K
    import tensorflow as tf
    
    def build():
        base = setting['base']
        P = len(setting['panels'])
        configs = Input(shape=(P,))
        configs_one_hot = K.one_hot(K.cast(configs,'int32'), width*height)
        matches = K.permute_dimensions(configs_one_hot, [0,2,1])
        matches = K.reshape(matches,[-1,P])
        panels = K.variable(setting['panels'])
        panels = K.reshape(panels, [P, base*base])
        states = tf.matmul(matches, panels)
        states = K.reshape(states, [-1, height, width, base, base])
        states = K.permute_dimensions(states, [0, 1, 3, 2, 4])
        states = K.reshape(states, [-1, height*base, width*base])
        return Model(configs, wrap(configs, states))

    model = build()
    return model.predict(np.array(configs),**kwargs)



def build_error(s, height, width, base):
    P = len(setting['panels'])
    s = K.reshape(s,[-1,height,base,width,base])
    s = K.permute_dimensions(s, [0,1,3,2,4])
    s = K.reshape(s,[-1,height,width,1,base,base])
    s = K.tile(s, [1,1,1,P,1,1,])
    
    allpanels = K.variable(np.array(setting['panels']))
    allpanels = K.reshape(allpanels, [1,1,1,P,base,base])
    allpanels = K.tile(allpanels, [K.shape(s)[0], height, width, 1, 1, 1])

    error = K.abs(s - allpanels)
    error = K.mean(error, axis=(4,5))
    return error

from .util import binary_search

def validate_states(states, verbose=True, **kwargs):
    base = setting['base']
    height = states.shape[1] // base
    width  = states.shape[2] // base
    load(width,height)
    
    if states.ndim == 4:
        assert states.shape[-1] == 1
        states = states[...,0]

    bs = binary_search(setting["min_threshold"],setting["max_threshold"])
    def build():
        states = Input(shape=(height*base,width*base))
        error = build_error(states, height, width, base)
        matches = K.cast(K.less_equal(error, bs.value), 'float32')
        # a, h, w, panel
        
        num_matches = K.sum(matches, axis=3)
        panels_ok = K.all(K.equal(num_matches, 1), (1,2))
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2))
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2))

        panel_coverage = K.sum(matches,axis=(1,2))
        # ideally, this should be [[1,1,1,1,1,1,1,1,1], ...]
        coverage_ok = K.all(K.less_equal(panel_coverage, 1), 1)
        coverage_ng = K.any(K.greater(panel_coverage, 1), 1)
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

    return validity


def to_configs(states, verbose=True, **kwargs):
    base = setting['base']
    width  = states.shape[1] // base
    height = states.shape[1] // base
    load(width,height)

    if states.ndim == 4:
        assert states.shape[-1] == 1
        states = states[...,0]

    def build():
        P = len(setting['panels'])
        states = Input(shape=(height*base,width*base))
        error = build_error(states, height, width, base)
        matches = 1 - K.clip(K.sign(error - setting["threshold"]),0,1)
        # a, h, w, panel

        config = K.reshape(matches, [K.shape(states)[0], height * width, -1])
        # a, pos, panel
        config = K.permute_dimensions(config, [0,2,1])
        # a, panel, pos
        config = config * K.arange(height*width,dtype='float')
        config = K.sum(config, axis=-1)

        num_matches = K.sum(matches, axis=3)
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2))
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2))

        return Model(states,
                     [ wrap(states, x) for x in [config,
                                                 panels_nomatch,
                                                 panels_ambiguous]])
    
    bs = binary_search(setting["min_threshold"],setting["max_threshold"])
    setting["threshold"] = bs.value
    while True:
        model = build()
        config, panels_nomatch, panels_ambiguous = model.predict(states, **kwargs)
        panels_nomatch = np.count_nonzero(panels_nomatch)
        panels_ambiguous = np.count_nonzero(panels_ambiguous)
        if verbose:
            print(f"threshold value: {bs.value}")
            print(panels_nomatch,                    "images have some panels which are unlike any panels")
            print(panels_ambiguous,                  "images have some panels which match >2 panels")
        if np.abs(panels_nomatch - panels_ambiguous) <= 1:
            return config
        elif panels_nomatch < panels_ambiguous:
            setting["threshold"] = bs.goleft()
        else:
            setting["threshold"] = bs.goright()

    return config


def states(width, height, configs=None, **kwargs):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height, **kwargs)

# old definition, slow
def transitions_old(width, height, configs=None, one_per_state=False):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        transitions = np.array([
            generate(
                [c1,random.choice(successors(c1,width,height))],width,height)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],width,height)
                                 for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)

def transitions(width, height, configs=None, one_per_state=False, **kwargs):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        pre = generate(configs, width, height, **kwargs)
        suc = generate(np.array([random.choice(successors(c1,width,height)) for c1 in configs ]), width, height, **kwargs)
        return np.array([pre, suc])
    else:
        transitions = np.array([ [c1,c2] for c1 in configs for c2 in successors(c1,width,height) ])
        pre = generate(transitions[:,0,:],width,height, **kwargs)
        suc = generate(transitions[:,1,:],width,height, **kwargs)
        return np.array([pre, suc])

def generate_configs(digit=9):
    import itertools
    return itertools.permutations(range(digit))

def generate_random_configs(digit=9,sample=10000):
    results = np.zeros((sample,digit))
    for i in range(sample):
        results[i] = np.random.permutation(digit)
    return results

def successors(config,width,height):
    pos = config[0]
    x = pos % width
    y = pos // width
    succ = []
    try:
        if x != 0:
            dir=1
            c = list(config)
            other = next(i for i,_pos in enumerate(c) if _pos == pos-1)
            c[0] -= 1
            c[other] += 1
            succ.append(c)
        if x != width-1:
            dir=2
            c = list(config)
            other = next(i for i,_pos in enumerate(c) if _pos == pos+1)
            c[0] += 1
            c[other] -= 1
            succ.append(c)
        if y != 0:
            dir=3
            c = list(config)
            other = next(i for i,_pos in enumerate(c) if _pos == pos-width)
            c[0] -= width
            c[other] += width
            succ.append(c)
        if y != height-1:
            dir=4
            c = list(config)
            other = next(i for i,_pos in enumerate(c) if _pos == pos+width)
            c[0] += width
            c[other] -= width
            succ.append(c)
        return succ
    except StopIteration:
        board = np.zeros((height,width))
        for i in range(height*width):
            _pos = config[i]
            _x = _pos % width
            _y = _pos // width
            board[_y,_x] = i
        print(board)
        print(succ)
        print(dir)
        print((c,x,y,width,height))

def validate_transitions_cpu_old(transitions, **kwargs):
    pre = np.array(transitions[0])
    suc = np.array(transitions[1])
    base = setting['base']
    height = pre.shape[1] // base
    width  = pre.shape[2] // base
    load(width,height)

    pre_validation = validate_states(pre, **kwargs)
    suc_validation = validate_states(suc, **kwargs)

    results = []
    for pre, suc, pre_validation, suc_validation in zip(pre, suc, pre_validation, suc_validation):
        
        if pre_validation and suc_validation:
            c = to_configs(np.array([pre, suc]), verbose=False)
            succs = successors(c[0], width, height)
            results.append(np.any(np.all(np.equal(succs, c[1]), axis=1)))
        else:
            results.append(False)
    
    return results

def validate_transitions_cpu(transitions, check_states=True, **kwargs):
    pre = np.array(transitions[0])
    suc = np.array(transitions[1])
    base = setting['base']
    height = pre.shape[1] // base
    width  = pre.shape[2] // base
    load(width,height)

    if check_states:
        pre_validation = validate_states(pre, verbose=False, **kwargs)
        suc_validation = validate_states(suc, verbose=False, **kwargs)

    pre_configs = to_configs(pre, verbose=False, **kwargs)
    suc_configs = to_configs(suc, verbose=False, **kwargs)
    
    results = []
    if check_states:
        for pre_c, suc_c, pre_validation, suc_validation in zip(pre_configs, suc_configs, pre_validation, suc_validation):

            if pre_validation and suc_validation:
                succs = successors(pre_c, width, height)
                results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
            else:
                results.append(False)
    else:
        for pre_c, suc_c in zip(pre_configs, suc_configs):
            succs = successors(pre_c, width, height)
            results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
    return results


validate_transitions = validate_transitions_cpu

# def to_objects(configs,width,height):
#     configs = np.array(configs)
#     xy = np.concatenate((np.expand_dims( np.array( configs  % 3, np.uint8),-1),
#                          np.expand_dims( np.array( configs // 3, np.uint8),-1)),
#                         axis=-1)
# 
#     return np.unpackbits(xy, 2)

# experimental
def to_objects(configs,width,height,shuffle=False):
    configs = np.array(configs)
    ix = np.eye(width)
    iy = np.eye(height)
    x = ix[np.array( configs  % 3, np.uint8)]
    y = iy[np.array( configs // 3, np.uint8)]

    # panels
    p = np.tile(np.eye(width*height), (len(configs),1,1))

    objects = np.concatenate((p,x,y), axis=-1)

    if shuffle:
        for sample in objects:
            np.random.shuffle(sample)

    return objects

def object_transitions(width, height, configs=None, one_per_state=False,shuffle=False, **kwargs):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        pre = to_objects(configs, width, height, shuffle)
        suc = to_objects(np.array([random.choice(successors(c1,width,height)) for c1 in configs ]), width, height, shuffle, **kwargs)
        return np.array([pre, suc])
    else:
        transitions = np.array([ [c1,c2] for c1 in configs for c2 in successors(c1,width,height) ])
        pre = to_objects(transitions[:,0,:],width,height, shuffle, **kwargs)
        suc = to_objects(transitions[:,1,:],width,height, shuffle, **kwargs)
        return np.array([pre, suc])

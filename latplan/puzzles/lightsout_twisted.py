#!/usr/bin/env python3

import random
import numpy as np
from .model.lightsout import generate_configs, generate_random_configs, successors
from .util import preprocess
from .util import wrap

from keras.layers import Input, Reshape, Cropping2D, MaxPooling2D
from keras.models import Model
import keras.backend.tensorflow_backend as K
import tensorflow as tf

panels = np.zeros((2,9,9))
panels[0, 4:6, :] = 1
panels[0, :, 4:6] = 1
pad = 1
relative_swirl_radius = 0.75
viscosity_adjustment = 0.18

swirl_args   = {'rotation':0, 'strength':3,  'preserve_range': True, 'order' : 1}
unswirl_args = {'rotation':0, 'strength':-3, 'preserve_range': True, 'order' : 1}

threshold = 0.04

def setup():
    pass

# from skimage.transform import swirl
# translate skimage.transform.swirl to tensorflow

def swirl_mapping(x, y, center, rotation, strength, radius):
    x0, y0 = center
    rho = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    radius = radius / 5 * np.log(2)

    theta = rotation + strength * \
            np.exp(-rho / radius) + \
            np.arctan2(y - y0, x - x0)

    return \
        x0 + rho * np.cos(theta), \
        y0 + rho * np.sin(theta)

# linear interpolation for batch tensor
def tensor_linear_interpolation(image, x, y, cval): # image: batch tensor, x,y: number
    import math
    x0, x1 = math.floor(x), math.ceil(x)
    y0, y1 = math.floor(y), math.ceil(y)
    dx0, dx1 = x - x0, x1 - x
    dy0, dy1 = y - y0, y1 - y

    shape = K.int_shape(image)[1:]
    results = []
    if 0 <= y0 and y0 < shape[0] and 0 <= x0 and x0 < shape[1]:
        results.append((y0,x0,dy1*dx1))
    
    if 0 <= y0 and y0 < shape[0] and 0 <= x1 and x1 < shape[1]:
        results.append((y0,x1,dy1*dx0))
    
    if 0 <= y1 and y1 < shape[0] and 0 <= x0 and x0 < shape[1]:
        results.append((y1,x0,dy0*dx1))
    
    if 0 <= y1 and y1 < shape[0] and 0 <= x1 and x1 < shape[1]:
        results.append((y1,x1,dy0*dx0))
    
    return results

def tensor_swirl(image, center=None, strength=1, radius=100, rotation=0, cval=0.0, **kwargs):
    # **kwargs is for unsupported options (ignored)
    cval = tf.fill(K.shape(image)[0:1], cval)
    shape = K.int_shape(image)[1:3]
    if center is None:
        center = np.array(shape) / 2
    ys = np.expand_dims(np.repeat(np.arange(shape[0]), shape[1]),-1)
    xs = np.expand_dims(np.tile  (np.arange(shape[1]), shape[0]),-1)
    map_xs, map_ys = swirl_mapping(xs, ys, center, rotation, strength, radius)

    mapping = np.zeros((*shape, *shape))
    for map_x, map_y, x, y in zip(map_xs, map_ys, xs, ys):
        results = tensor_linear_interpolation(image, map_x, map_y, cval)
        for _y, _x, w in results:
            # mapping[int(y),int(x),int(_y),int(_x),] = w
            mapping[int(_y),int(_x),int(y),int(x),] = w

    
    results = tf.tensordot(image, K.variable(mapping), [[1,2],[0,1]])
    # results = K.reshape(results, K.shape(image))
    return results

def batch_swirl(images):
    from skimage.transform import swirl
    images = np.array(images)
    r = images.shape[1] * relative_swirl_radius
    # from joblib import Parallel, delayed
    # return np.array(Parallel(n_jobs=4)(delayed(swirl)(i, radius=r, **swirl_args) for i in images))
    return np.array([ swirl(i, radius=r, **swirl_args) for i in images ])

def batch_unswirl(images):
    from skimage.transform import swirl
    images = np.array(images)
    r = images.shape[1] * relative_swirl_radius
    # from joblib import Parallel, delayed
    # return np.array(Parallel(n_jobs=4)(delayed(swirl)(i, radius=r, **unswirl_args) for i in images))
    return np.array([ swirl(i, radius=r, **unswirl_args) for i in images ])

def generate_cpu(configs, **kwargs):
    configs = np.array(configs)
    import math
    size = int(math.sqrt(len(configs[0])))
    base = panels.shape[1]
    dim = base*size
    def generate(config):
        figure_big = np.zeros((dim+2*pad,dim+2*pad))
        figure = figure_big[pad:-pad,pad:-pad]
        for pos,value in enumerate(config):
            x = pos % size
            y = pos // size
            if value > 0:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = panels[0]
            else:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = panels[1]
        
        return figure_big
    return preprocess(batch_swirl([ generate(c) for c in configs ]))

def generate_gpu(configs,**kwargs):
    configs = np.array(configs)
    import math
    size = int(math.sqrt(len(configs[0])))
    base = panels.shape[1]
    dim = base*size

    def build():
        P = 2
        configs = Input(shape=(size*size,))
        _configs = 1 - K.round((configs/2)+0.5) # from -1/1 to 1/0
        configs_one_hot = K.one_hot(K.cast(_configs,'int32'), P)
        configs_one_hot = K.reshape(configs_one_hot, [-1,P])
        _panels = K.variable(panels)
        _panels = K.reshape(_panels, [P, base*base])
        states = tf.matmul(configs_one_hot, _panels)
        states = K.reshape(states, [-1, size, size, base, base])
        states = K.permute_dimensions(states, [0, 1, 3, 2, 4])
        states = K.reshape(states, [-1, size*base, size*base, 1])
        states = K.spatial_2d_padding(states, padding=((pad,pad),(pad,pad)))
        states = K.squeeze(states, -1)
        return Model(configs, wrap(configs, states))

    return preprocess(batch_swirl(build().predict(configs,**kwargs)))

def generate_gpu2(configs,**kwargs):
    configs = np.array(configs)
    import math
    size = int(math.sqrt(len(configs[0])))
    base = panels.shape[1]
    dim = base*size

    def build():
        P = 2
        configs = Input(shape=(size*size,))
        _configs = 1 - K.round((configs/2)+0.5) # from -1/1 to 1/0
        configs_one_hot = K.one_hot(K.cast(_configs,'int32'), P)
        configs_one_hot = K.reshape(configs_one_hot, [-1,P])
        _panels = K.variable(panels)
        _panels = K.reshape(_panels, [P, base*base])
        states = tf.matmul(configs_one_hot, _panels)
        states = K.reshape(states, [-1, size, size, base, base])
        states = K.permute_dimensions(states, [0, 1, 3, 2, 4])
        states = K.reshape(states, [-1, size*base, size*base, 1])
        states = K.spatial_2d_padding(states, padding=((pad,pad),(pad,pad)))
        states = K.squeeze(states, -1)
        states = tensor_swirl(states, radius=dim+2*pad * relative_swirl_radius, **swirl_args)
        return Model(configs, wrap(configs, states))

    return preprocess(build().predict(configs,**kwargs))

generate = generate_gpu2

def states(size, configs=None, **kwargs):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs, **kwargs)

def transitions_old(size, configs=None, one_per_state=False, **kwargs):
    if configs is None:
        configs = generate_configs(size)
    if one_per_state:
        transitions = np.array([
            generate([c1,random.choice(successors(c1))], **kwargs)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2], **kwargs)
                                 for c1 in configs for c2 in successors(c1) ])
    return np.einsum('ab...->ba...',transitions)

def transitions(size, configs=None, one_per_state=False, **kwargs):
    if configs is None:
        configs = generate_configs(size)
    if one_per_state:
        pre = generate(configs, **kwargs)
        suc = generate(np.array([random.choice(successors(c1)) for c1 in configs ]), **kwargs)
        return np.array([pre, suc])
    else:
        transitions = np.array([ [c1,c2] for c1 in configs for c2 in successors(c1) ])
        pre = generate(transitions[:,0,:], **kwargs)
        suc = generate(transitions[:,1,:], **kwargs)
        return np.array([pre, suc])

def build_errors(states,base,pad,dim,size):
    # address the numerical viscosity in swirling
    s = K.round(states+viscosity_adjustment)
    s = Reshape((dim+2*pad,dim+2*pad,1))(s)
    s = Cropping2D(((pad,pad),(pad,pad)))(s)
    s = K.reshape(s,[-1,size,base,size,base])
    s = K.permute_dimensions(s, [0,1,3,2,4])
    s = K.reshape(s,[-1,size,size,1,base,base])
    s = K.tile   (s,[1, 1, 1, 2, 1, 1,]) # number of panels : 2

    allpanels = K.variable(panels)
    allpanels = K.reshape(allpanels, [1,1,1,2,base,base])
    allpanels = K.tile(allpanels, [K.shape(s)[0], size,size, 1, 1, 1])
    
    def hash(x):
        ## 2x2 average hashing
        x = K.reshape(x, [-1,size,size,2, base//3, 3, base//3, 3])
        x = K.mean(x, axis=(5,7))
        return K.round(x)
        ## diff hashing (horizontal diff)
        # x1 = x[:,:,:,:,:,:-1]
        # x2 = x[:,:,:,:,:,1:]
        # d = x1 - x2
        # return K.round(d)
        ## just rounding
        # return K.round(x)
        ## do nothing
        # return x

    # s         = hash(s)
    # allpanels = hash(allpanels)
    
    # error = K.binary_crossentropy(s, allpanels)
    error = K.abs(s - allpanels)
    error = hash(error)
    error = K.mean(error, axis=(4,5))
    return error

def validate_states(states,verbose=True,**kwargs):
    base = panels.shape[1]
    dim  = states.shape[1] - pad*2
    size = dim // base
    
    if states.ndim == 4:
        assert states.shape[-1] == 1
        states = states[...,0]

    def build():
        states = Input(shape=(dim+2*pad,dim+2*pad))
        s = tensor_swirl(states, radius=dim+2*pad * relative_swirl_radius, **unswirl_args)
        error = build_errors(s,base,pad,dim,size)
        matches = 1 - K.clip(K.sign(error - threshold),0,1)
        num_matches = K.sum(matches, axis=3)
        panels_ok = K.all(K.equal(num_matches, 1), (1,2))
        panels_ng = K.any(K.not_equal(num_matches, 1), (1,2))
        panels_nomatch   = K.any(K.equal(num_matches, 0), (1,2))
        panels_ambiguous = K.any(K.greater(num_matches, 1), (1,2))

        validity = panels_ok
        
        if verbose:
            return Model(states,
                         [ wrap(states, x) for x in [panels_ng,
                                                     panels_nomatch,
                                                     panels_ambiguous,
                                                     validity]])
        else:
            return Model(states, wrap(states, validity))
    
    if verbose:
        panels_ng, panels_nomatch, panels_ambiguous, validity \
            = build().predict(states, **kwargs)
        print(np.count_nonzero(panels_ng),       "images have some panels which match 0 or >2 panels, out of which")
        print(np.count_nonzero(panels_nomatch),  "images have some panels which are unlike any panels")
        print(np.count_nonzero(panels_ambiguous),"images have some panels which match >2 panels")
        print(np.count_nonzero(validity),        "images have panels (all of them) which match exactly 1 panel each")
        return validity
    else:
        validity \
            = build().predict(states, **kwargs)
        return validity


def to_configs(states, verbose=True, **kwargs):
    base = panels.shape[1]
    dim  = states.shape[1] - pad*2
    size = dim // base

    if states.ndim == 4:
        assert states.shape[-1] == 1
        states = states[...,0]
    
    def build():
        states = Input(shape=(dim+2*pad,dim+2*pad))
        s = tensor_swirl(states, radius=dim+2*pad * relative_swirl_radius, **unswirl_args)
        error = build_errors(s,base,pad,dim,size)
        matches = 1 - K.clip(K.sign(error - threshold),0,1)
        # a, h, w, panel
        matches = K.reshape(matches, [K.shape(states)[0], size * size, -1])
        # a, pos, panel
        config = matches * K.arange(2,dtype='float')
        config = K.sum(config, axis=-1)
        # this is 0,1 configs; for compatibility, we need -1 and 1
        config = - (config - 0.5)*2
        return Model(states, wrap(states, K.round(config)))
    
    return build().predict(states, **kwargs)


def validate_transitions(transitions, check_states=True, **kwargs):
    pre = np.array(transitions[0])
    suc = np.array(transitions[1])

    if check_states:
        pre_validation = validate_states(pre, verbose=False, **kwargs)
        suc_validation = validate_states(suc, verbose=False, **kwargs)

    pre_configs = to_configs(pre, verbose=False, **kwargs)
    suc_configs = to_configs(suc, verbose=False, **kwargs)
    
    results = []
    if check_states:
        for pre_c, suc_c, pre_validation, suc_validation in zip(pre_configs, suc_configs, pre_validation, suc_validation):

            if pre_validation and suc_validation:
                succs = successors(pre_c)
                results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
            else:
                results.append(False)
    else:
        for pre_c, suc_c in zip(pre_configs, suc_configs):
            succs = successors(pre_c)
            results.append(np.any(np.all(np.equal(succs, suc_c), axis=1)))
    return results

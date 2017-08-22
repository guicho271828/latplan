#!/usr/bin/env python3

import numpy as np
from .model.lightsout import generate_configs, generate_random_configs, successors
from .util import wrap
from .util import preprocess

buttons = [
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 1, 0, 0, 0, 0, ],],
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, ],]
]

def setup():
    pass

def generate_cpu(configs):
    import math
    size = int(math.sqrt(len(configs[0])))
    base = 9
    dim = base*size
    half = dim//2
    def generate(config):
        figure = np.zeros((dim,dim))
        for pos,value in enumerate(config):
            x = pos % size
            y = pos // size
            if value > 0:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = buttons[0]
            else:
                figure[y*base:(y+1)*base,
                       x*base:(x+1)*base] = buttons[1]
        # np.skew or np.XXX or any visual effect
        figure2 = np.zeros((dim*4,dim*4))
        figure2[dim+half:2*dim+half,
                dim+half:2*dim+half] = figure
        figure3 = np.zeros((dim*2,dim*2))
        for x in range(-dim,dim):
            for y in range(-dim,dim):
                r = math.sqrt(x*x+y*y)
                rad = math.atan2(y,x) + math.pi  * (1-r)/half/5
                p = r * np.array([math.cos(rad), math.sin(rad)])
                px1 = math.floor(p[0])
                py1 = math.floor(p[1])
                grid = np.array([[[px1,py1],[px1+1,py1],],
                                 [[px1,py1+1],[px1+1,py1+1],],])
                w = (1 - np.prod(np.fabs(grid - p),axis=-1))/3
                
                value = np.sum(w * figure2[py1+2*dim:py1+2+2*dim,px1+2*dim:px1+2+2*dim])
                figure3[y+dim,x+dim] = value*2
        return preprocess(figure3)
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim*2,dim*2)).clip(0,1)


def generate_gpu(configs,**kwargs):
    import math
    configs = np.array(configs)
    print(configs)
    size = int(math.sqrt(len(configs[0])))
    base = 9
    dim = base*size
    half = dim//2
    
    from keras.layers import Input, Reshape
    from keras.models import Model
    from keras import backend as K
    import tensorflow as tf

    def build():
        P = 2
        configs = Input(shape=(size*size,)) # configs are -1/1 array
        c = configs
        c = (c + 1)/2 + 1            # now this takes the value of 1 or 2
        c_one_hot = K.one_hot(K.cast(c,'int32'), P)
        c_one_hot = wrap(configs, c_one_hot)
        matches = K.permute_dimensions(c_one_hot, [0,2,1])
        matches = K.reshape(matches,[-1,P])
        panels = K.variable(buttons)
        panels = K.reshape(panels, [P, base*base])
        states = tf.matmul(matches, panels)
        states = K.reshape(states, [-1, size, size, base, base])
        states = K.permute_dimensions(states, [0, 1, 3, 2, 4])
        states = K.reshape(states, [-1, size*base, size*base])
        states = wrap(c_one_hot, states)
        # whirl effect

        # cartesian to polar conversion matrix
        from math import cos, sin, ceil, floor, pi, sqrt, atan2
        m = np.zeros((360, half, dim, dim))
        for deg in range(360):
            th = deg * 2 * pi / 360
            for r in range(half):
                x, y = r*cos(th) + half, r*sin(th) + half
                x0, x1 = floor(x), ceil(x)
                y0, y1 = floor(y), ceil(y)
                grid = np.array([[[x0,y0],[x1,y0]],
                                 [[x0,y1],[x1,y1]]])
                w = (1-np.prod(np.abs(grid - [x,y]), axis=-1)) / 3
                m[deg,r,x0:x0+2,y0:y0+2] = w
        M = K.variable(m)
        polar = tf.tensordot(states, M, [[1,2],[2,3]])
        polar = wrap(states, polar)
        
        m2 = np.zeros((360, half, 360, half))
        # swirl degree at the edge of the image
        swirl_deg = 60
        for deg in range(360):
            for r in range(half):
                d = deg + r/half * swirl_deg
                d0, d1 = floor(d), ceil(d)
                m2[deg,r,d0%360,r] = d1-d
                m2[deg,r,d1%360,r] = d-d0
        M2 = K.variable(m2)
        swirled_polar = tf.tensordot(polar, M2, [[1,2],[2,3]])
        swirled_polar = wrap(polar, swirled_polar)

        # polar to cartesian
        m3 = np.zeros((dim, dim, 360, half))
        for y in range(dim):
            if y == half:
                continue
            for x in range(dim):
                if x == half:
                    continue
                r  = sqrt((x - half)**2+(y - half)**2)
                th = atan2((y - half),(x - half))
                if th < 0:
                    th += 2 * pi
                d = th * 360 / (2 * pi)
                
                d0, d1 = floor(d), ceil(d)
                r0, r1 = floor(r), ceil(r)
                grid = np.array([[[d0,r0],[d1,r0]],
                                 [[d0,r1],[d1,r1]]])
                w = (1-np.prod(np.abs(grid - [d,r]), axis=-1)) / 3
                # print(w.shape,y,x,th,d,d0,r,r0,r1,half)
                if d1 < 360 and r1 < half:
                    m3[y,x,d0:d0+2,r0:r0+2] = w
        M3 = K.variable(m3)
        swirled = tf.tensordot(swirled_polar, M3, [[1,2],[2,3]])
        swirled = wrap(swirled_polar, swirled)
        
        return Model(configs, swirled)
    model = build()
    # model.summary()
    return model.predict(configs,**kwargs)

generate = generate_cpu

def states(size, configs=None):
    if configs is None:
        configs = generate_configs(size)
    return generate(configs)

def transitions(size, configs=None, one_per_state=False):
    if configs is None:
        configs = generate_configs(size)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate([c1,pickone(successors(c1))])
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2])
                                 for c1 in configs for c2 in successors(c1) ])
    return np.einsum('ab...->ba...',transitions)



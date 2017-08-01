#!/usr/bin/env python3

import numpy as np

setting = {
    'base' : None,
    'panels' : None,
    'loader' : None,
}

def load(width,height,force=False):
    if setting['panels'] is None or force is True:
        setting['panels'] = setting['loader'](width,height)

def generate(configs, width, height):
    assert width*height <= 9
    load(width, height)
    dim_x = setting['base']*width
    dim_y = setting['base']*height
    def generate(config):
        figure = np.zeros((dim_y,dim_x))
        for digit,pos in enumerate(config):
            x = pos % width
            y = pos // width
            figure[y*setting['base']:(y+1)*setting['base'],
                   x*setting['base']:(x+1)*setting['base']] = setting['panels'][digit]
        return figure
    return np.array([ generate(c) for c in configs ]).reshape((-1,dim_y,dim_x))

def validate_states(states, width, height, verbose=True):
    load(width, height)
    base = setting['base']
    
    states = np.einsum("ahywx->ahwyx",
                       np.reshape(states.round(),
                                  [-1,height,base,width,base]))
    if verbose:
        print(states.shape)

    panels = np.array(setting['panels'])
    if verbose:
        print(panels.shape)

    matches = np.zeros((len(states), height, width, len(panels)),dtype=np.int8)
    if verbose:
        print(matches.shape)

    abs = states.copy()
    mae = np.zeros((len(states), height, width))
    for i, panel in enumerate(panels):
        if verbose:
            print(".",end="",flush=True)
        np.absolute(states - panel, out=abs)
        np.mean(abs, axis=(3,4), out=mae)
        matches[(*np.where(mae < 0.01),i)] = 1

    num_matches = np.sum(matches, axis=3)
    if verbose:
        print(num_matches.shape)

    panels_ok = np.all(num_matches == 1, (1,2))
    panels_ng = np.any(num_matches != 1, (1,2))
    panels_nomatch   = np.any(num_matches == 0, (1,2))
    panels_ambiguous = np.any(num_matches >  1, (1,2))
    
    if verbose:
        print(np.count_nonzero(panels_ng),       "images have some panels which match 0 or >2 panels, out of which")
        print(np.count_nonzero(panels_nomatch),  "images have some panels which are unlike any panels")
        print(np.count_nonzero(panels_ambiguous),"images have some panels which match >2 panels")
        print(np.count_nonzero(panels_ok),       "images have panels (all of them) which match exactly 1 panel each")

    panel_coverage = np.sum(matches,axis=(1,2))
    if verbose:
        print(panel_coverage.shape)
    # ideally, this should be [[1,1,1,1,1,1,1,1,1], ...]
    coverage_ok = np.all(panel_coverage <= 1, 1)
    coverage_ng = np.any(panel_coverage >  1, 1)
    
    if verbose:
        print(np.count_nonzero(np.logical_and(panels_ok, coverage_ng)),"images have duplicated tiles")
        print(np.count_nonzero(np.logical_and(panels_ok, coverage_ok)),"images have no duplicated tiles")

    return np.logical_and(panels_ok, coverage_ok)


def to_configs(states, width, height, verbose=True):
    load(width, height)
    base = setting['base']
    
    states = np.einsum("ahywx->ahwyx",
                       np.reshape(states.round(),
                                  [-1,height,base,width,base]))

    panels = np.array(setting['panels'])

    matches = np.zeros((len(states), height, width, len(panels)),dtype=np.int8)
    if verbose:
        print(matches.shape)

    abs = states.copy()
    mae = np.zeros((len(states), height, width))
    for i, panel in enumerate(panels):
        if verbose:
            print(".",end="",flush=True)
        np.absolute(states - panel, out=abs)
        np.mean(abs, axis=(3,4), out=mae)
        matches[(*np.where(mae < 0.01),i)] = 1

    configs = np.zeros((len(matches), height*width))
    npos, vpos, hpos, ppos = np.where(matches == 1)
    configs[npos,ppos] = vpos * height + hpos
    return configs

def states(width, height, configs=None):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    return generate(configs,width,height)

def transitions(width, height, configs=None, one_per_state=False):
    digit = width * height
    if configs is None:
        configs = generate_configs(digit)
    if one_per_state:
        def pickone(thing):
            index = np.random.randint(0,len(thing))
            return thing[index]
        transitions = np.array([
            generate(
                [c1,pickone(successors(c1,width,height))],width,height)
            for c1 in configs ])
    else:
        transitions = np.array([ generate([c1,c2],width,height)
                                 for c1 in configs for c2 in successors(c1,width,height) ])
    return np.einsum('ab...->ba...',transitions)

def generate_configs(digit=9):
    import itertools
    return itertools.permutations(range(digit))

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


def validate_transitions(transitions, width, height):
    pre = transitions[0]
    suc = transitions[1]

    pre_validation = validate_states(pre, width, height)
    suc_validation = validate_states(suc, width, height)

    results = []
    for pre, suc, pre_validation, suc_validation in zip(pre, suc, pre_validation, suc_validation):
        
        if pre_validation and suc_validation:
            c = to_configs(np.array([pre, suc]), width, height, verbose=False)
            succs = successors(c[0], width, height)
            results.append(np.any(np.all(np.equal(succs, c[1]), axis=1)))
        else:
            results.append(False)
    
    return results

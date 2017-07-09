#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
from model import default_networks

import keras.backend as K
import tensorflow as tf

import strips
from strips import modes
from strips import dump, dump_all_actions, run
from plan   import options, latent_plan

float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def random_walk(init_c,length,successor_fn):
    current = init_c
    trace = [current]
    for i in range(length):
        sucs = successor_fn(current)
        suc = sucs[random.randint(len(sucs))]
        trace.append(suc)
        current = suc
    return np.array(trace)

def mnist_puzzle():
    strips.parameters = [[4000],[0.4],[49]]
    strips.epoch = 1000
    strips.batch_size = 2000
    print(strips.parameters,strips.epoch,strips.batch_size)

    import puzzles.mnist_puzzle as p
    def convert(panels):
        return np.array([
            [i for i,x in enumerate(panels) if x == p]
            for p in range(9)]).reshape(-1)
    ig_c = [convert([8,0,6,5,4,7,2,3,1]),
            convert([0,1,2,3,4,5,6,7,8])]
    ig = p.states(3,3,ig_c)
    train_c = np.array([ random_walk(ig_c[0],300, lambda config: p.successors(config,3,3))
                         for i in range(40) ])
    train_c = train_c.reshape((-1,9))
    train = p.states(3,3,train_c)
    
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)

    test_c = configs[:1000]
    test = p.states(3,3,test_c)

    ae = run(learn_flag,"samples/mnist_puzzle33p_rw_40restarts_{}/".format(strips.encoder), train, test)
    dump(ae, train, test, p.transitions(3,3,train_c,True))
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    dump_all_actions(ae,configs,lambda configs: p.transitions(3,3,configs))
    latent_plan(*ig, ae, option)

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        print({ k for k in default_networks})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
        print({k for k in modes})
        print({k for k in options})
    else:
        print('args:',sys.argv)
        strips.encoder = sys.argv[1]
        if strips.encoder not in default_networks:
            raise ValueError("invalid encoder!: {}".format(sys.argv))
        task = sys.argv[2]
        mode = sys.argv[3]
        if mode not in modes:
            raise ValueError("invalid mode!: {}".format(sys.argv))
        learn_flag = modes[mode]
        option = sys.argv[4]
        if option not in options:
            raise ValueError("invalid option!: {}".format(sys.argv))
        globals()[task]()

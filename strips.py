#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
from latplan.model import default_networks
from latplan.util        import curry
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf


float_formatter = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

encoder = 'fc'
mode = 'learn_dump'
modes = {'learn':True,'learn_dump':True,'dump':False}
learn_flag = True

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 5.0,
    'min_temperature' : 0.7,
    'M'               : 2,
}

def dump_autoencoding_image(ae,test,train):
    rz = np.random.randint(0,2,(6,ae.parameters['N']))
    ae.plot_autodecode(rz,"autodecoding_random.png",verbose=True)
    ae.plot(select(test,12),"autoencoding_test.png",verbose=True)
    ae.plot(select(train,12),"autoencoding_train.png",verbose=True)
    
def dump_all_actions(ae,configs,trans_fn,name="all_actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 10000
    loop = (l // batch) + 1
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            for i in range(repeat):
                for begin in range(0,loop*batch,batch):
                    end = begin + batch
                    print((begin,end,len(configs)))
                    transitions = trans_fn(configs[begin:end])
                    orig, dest = transitions[0], transitions[1]
                    orig_b = ae.encode_binary(orig,batch_size=1000).round().astype(int)
                    dest_b = ae.encode_binary(dest,batch_size=1000).round().astype(int)
                    actions = np.concatenate((orig_b,dest_b), axis=1)
                    np.savetxt(f,actions,"%d")
    except AttributeError:
        print("this AE does not support dumping")
    except KeyboardInterrupt:
        print("dump stopped")

def dump_all_states(ae,configs,states_fn,name="all_states.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 10000
    loop = (l // batch) + 1
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            for i in range(repeat):
                for begin in range(0,loop*batch,batch):
                    end = begin + batch
                    print((begin,end,len(configs)))
                    states = states_fn(configs[begin:end])
                    states_b = ae.encode_binary(states,batch_size=1000).round().astype(int)
                    np.savetxt(f,states_b,"%d")
    except AttributeError:
        print("this AE does not support dumping")
    except KeyboardInterrupt:
        print("dump stopped")

################################################################

# note: lightsout has epoch 200

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]


def run(learn,path,train,test):
    if learn:
        from latplan.util import curry
        ae, _, _ = grid_search(curry(nn_task, default_networks[encoder], path,
                                     train, train, test, test),
                               default_parameters,
                               parameters,
                               report = lambda ae: dump_autoencoding_image(ae,test,train))
    else:
        ae = default_networks[encoder](path).load()
        ae.summary()
    return ae

parameters = {
    'layer'      :[2000],# [400,4000],
    'clayer'     :[16],# [400,4000],
    'dropout'    :[0.4], #[0.1,0.4],
    'N'          :[36],  #[25,49],
    'dropout_z'  :[False],
    'activation' :['tanh'],
    'full_epoch' :[1000],
    'epoch'      :[1000],
    'batch_size' :[2000],
    'lr'         :[0.001],
}

def puzzle_mnist(width=3,height=3):
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/puzzle_mnist{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)
    dump_all_states(ae,configs[:13000],lambda configs: p.states(width,height,configs),"states.csv")
    dump_all_states(ae,configs,        lambda configs: p.states(width,height,configs),)

def puzzle_mnist_zdropout(width=3,height=3):
    # Made to see if increasing the z_dim and enabling the dropout
    # would solve the "invalid states problem". 
    # The purpose is to see if the randomly generated bitvector
    # decodes into a valid state and it encodes into itself.
    # Did not achieve the expected outcome. Perhaps reulating the
    # latent layer with infoGAN-like training is necessary.
    import latplan.puzzles.puzzle_mnist as p
    p.setup()
    global parameters
    parameters = {
        'layer'      :[2000],# [400,4000],
        'clayer'     :[16],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'N'          :[100],  #[25,49],
        'dropout_z'  :[True],
        'activation' : ['tanh'],
        'full_epoch' :[500],
        'epoch'      :[200],
        'batch_size' :[2000],
        'lr'         :[0.001],
    }
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/puzzle_mnist_zdropout{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)
    dump_all_states(ae,configs[:13000],lambda configs: p.states(width,height,configs),"states.csv")
    dump_all_states(ae,configs,        lambda configs: p.states(width,height,configs),)

def puzzle_lenna(width=3,height=3):
    import latplan.puzzles.puzzle_lenna as p
    p.setup()
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/puzzle_lenna{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)

def puzzle_mandrill(width=3,height=3):
    import latplan.puzzles.puzzle_mandrill as p
    p.setup()
    configs = p.generate_configs(width*height)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    train       = p.states(width,height,train_c)
    test        = p.states(width,height,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/puzzle_mandrill{}{}_{}/".format(width,height,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(width,height,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(width,height,configs),)

def hanoi(disks=5,towers=3):
    global parameters
    if 'fc' in encoder:
        parameters = {
            'layer'      :[2000],# [400,4000],
            'clayer'     :[9],# [400,4000],
            'dropout'    :[0.4], #[0.1,0.4],
            'N'          :[49],  #[25,49],
            'dropout_z'  :[False],
            'activation' : ['relu'],
            'full_epoch' :[1000],
            'epoch'      :[1000],
            'batch_size' :[500],
            'optimizer'  :['adam'],
            'lr'         :[0.001],
        }
    else:
        parameters = {
            'layer'      :[1000],# [400,4000],
            'clayer'     :[12],# [400,4000],
            'dropout'    :[0.4], #[0.1,0.4],
            'N'          :[49],  #[25,49],
            'dropout_z'  :[False],
            'activation' : ['relu'],
            'full_epoch' :[1000],
            'epoch'      :[1000],
            'batch_size' :[500],
            'optimizer'  :['adam'],
            'lr'         :[0.001],
        }
    import latplan.puzzles.hanoi as p
    configs = p.generate_configs(disks,towers)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    print(len(configs))
    configs = configs[:min(8000,len(configs))]
    print(len(configs))
    train_c = configs[:int(0.9*len(configs))]
    test_c  = configs[int(0.9*len(configs)):]
    train       = p.states(disks,towers,train_c)
    test        = p.states(disks,towers,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/hanoi{}{}_{}/".format(disks,towers,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,lambda configs: p.transitions(disks,towers,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(disks,towers,configs),)
    dump_all_states(ae,configs,lambda configs: p.states(disks,towers,configs),"states.csv")
    dump_all_states(ae,configs,        lambda configs: p.states(disks,towers,configs),)

def digital_lightsout(size=4):
    global parameters
    parameters = {
        'layer'      :[2000],# [400,4000],
        'clayer'     :[16],# [400,4000],
        'dropout'    :[0.4], #[0.1,0.4],
        'N'          :[28],  #[25,49],
        'dropout_z'  :[False],
        'activation' : ['tanh'],
        'full_epoch' :[1000],
        'epoch'      :[500],
        'batch_size' :[2000],
        'lr'         :[0.001],
    }
    import latplan.puzzles.digital_lightsout as p
    print('generating configs...')
    configs = p.generate_configs(size)
    random.shuffle(configs)
    train_c = configs[:12000]
    test_c  = configs[12000:13000]
    print('generating figures...')
    train       = p.states(size,train_c)
    test        = p.states(size,test_c)

    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/digital_lightsout_{}_{}/".format(size,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs[:13000],lambda configs: p.transitions(size,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs),)

def digital_lightsout_skewed(size=3):
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[49],
        'epoch'      :[1000],
        'batch_size' :[2000]
    }
    import latplan.puzzles.digital_lightsout_skewed as p
    print('generating configs...')
    configs = p.generate_configs(size)
    random.shuffle(configs)
    train_c = configs[:int(len(configs)*0.9)]
    test_c  = configs[int(len(configs)*0.9):]
    print('generating figures...')
    train       = p.states(size,train_c)
    test        = p.states(size,test_c)
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/digital_lightsout_skewed_{}_{}/".format(size,encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(size,configs))

def counter_mnist():
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[36],
        'epoch'      :[1000],
        'batch_size' :[3500]
    }
    import latplan.puzzles.counter_mnist as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/counter_mnist_{}/".format(encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs))

def counter_random_mnist():
    global parameters
    parameters = {
        'layer'      :[4000],
        'dropout'    :[0.4],
        'N'          :[36],
        'epoch'      :[1000],
        'batch_size' :[3500]
    }
    import latplan.puzzles.counter_random_mnist as p
    configs = np.repeat(p.generate_configs(10),10000,axis=0)
    states = p.states(10,configs)
    train       = states[:int(len(states)*(0.8))]
    test        = states[int(len(states)*(0.8)):]
    print(len(configs),len(train),len(test))
    ae = run(learn_flag,"samples/counter_random_mnist_{}/".format(encoder), train, test)
    dump_autoencoding_image(ae,test,train)
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs),"actions.csv")
    dump_all_actions(ae,configs,        lambda configs: p.transitions(10,configs))


def main():
    global encoder, mode, learn_flag
    import sys
    if len(sys.argv) == 1:
        print({ k for k in default_networks})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
        print({ k for k in modes})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        encoder = sys.argv.pop(0)
        if encoder not in default_networks:
            raise ValueError("invalid encoder!: {}".format(encoder))
        task = sys.argv.pop(0)
        mode = sys.argv.pop(0)
        if mode not in modes:
            raise ValueError("invalid mode!: {}".format(mode))
        learn_flag = modes[mode]
        globals()[task](*map(eval,sys.argv))
    
if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import config
import numpy as np
import numpy.random as random
import latplan
import latplan.model
from latplan.util        import curry
from latplan.util.tuning import *
from latplan.util.noise  import gaussian

import keras.backend as K
import tensorflow as tf

import os
import os.path

float_formatter = lambda x: "%.5f" % x
import sys
np.set_printoptions(formatter={'float_kind':float_formatter})

mode     = 'learn_dump'
sae_path = None

from keras.optimizers import Adam
from keras_adabound   import AdaBound
from keras_radam      import RAdam

import keras.optimizers

setattr(keras.optimizers,"radam", RAdam)
setattr(keras.optimizers,"adabound", AdaBound)

# default values
default_parameters = {
    'epoch'           : 200,
    'batch_size'      : 500,
    'optimizer'       : "radam",
    'max_temperature' : 5.0,
    'min_temperature' : 0.7,
    'M'               : 2,
    'train_gumbel'    : True,    # if true, noise is added during training
    'train_softmax'   : True,    # if true, latent output is continuous
    'test_gumbel'     : False,   # if true, noise is added during testing
    'test_softmax'    : False,   # if true, latent output is continuous
    'locality'        : 0.0,
    'locality_delay'  : 0.0,
}
# hyperparameter tuning
parameters = {
    'beta'       :[-0.3,-0.1,0.0,0.1,0.3],
    'lr'         :[0.1,0.01,0.001],
    'N'          :[100,200,500,1000],
    'M'          :[2],
    'layer'      :[1000],
    'clayer'     :[16],
    'dropout'    :[0.4],
    'noise'      :[0.4],
    'dropout_z'  :[False],
    'activation' :['relu'],
    'num_actions'    :[100,200,400,800,1600],
    'aae_width'      :[100,300,600,],
    'aae_depth'      :[0,1,2],
    'aae_activation' :['relu','tanh'],
    'aae_delay'      :[0,],
    'direct'             :[0.1,1.0,10.0],
    'direct_delay'       :[0.05,0.1,0.2,0.3,0.5],
    'zerosuppress'       :[0.1,0.2,0.5],
    'zerosuppress_delay' :[0.05,0.1,0.2,0.3,0.5],
    'loss'               :["BCE"],
}

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def plot_autoencoding_image(ae,test,train):
    if 'plot' not in mode:
        return
    rz = np.random.randint(0,2,(6,*ae.zdim()))
    ae.plot_autodecode(rz,ae.local("autodecoding_random.png"),verbose=True)
    ae.plot(test[:6],ae.local("autoencoding_test.png"),verbose=True)
    ae.plot(train[:6],ae.local("autoencoding_train.png"),verbose=True)

def dump_all_actions(ae,configs,trans_fn,name="all_actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                transitions = trans_fn(configs[begin:end])
                pre, suc = transitions[0], transitions[1]
                pre_b = ae.encode(pre,batch_size=1000).round().astype(int)
                suc_b = ae.encode(suc,batch_size=1000).round().astype(int)
                actions = np.concatenate((pre_b,suc_b), axis=1)
                np.savetxt(f,actions,"%d")

def dump_actions(ae,transitions,name="actions.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    pre, suc = transitions[0], transitions[1]
    if ae.parameters["test_gumbel"]:
        pre = np.repeat(pre,axis=0,repeats=10)
        suc = np.repeat(suc,axis=0,repeats=10)
    pre = ae.encode(pre,batch_size=1000)
    suc = ae.encode(suc,batch_size=1000)
    ae.dump_actions(pre,suc,batch_size=1000)

def dump_all_states(ae,configs,states_fn,name="all_states.csv",repeat=1):
    if 'dump' not in mode:
        return
    l = len(configs)
    batch = 5000
    loop = (l // batch) + 1
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            for begin in range(0,loop*batch,batch):
                end = begin + batch
                print((begin,end,len(configs)))
                states = states_fn(configs[begin:end])
                states_b = ae.encode(states,batch_size=1000).round().astype(int)
                np.savetxt(f,states_b,"%d")

def dump_states(ae,states,name="states.csv",repeat=1):
    if 'dump' not in mode:
        return
    print(ae.local(name))
    with open(ae.local(name), 'wb') as f:
        for i in range(repeat):
            np.savetxt(f,ae.encode(states,batch_size=1000).round().astype(int),"%d")

################################################################

# note: lightsout has epoch 200

def run(path,train,val,parameters,train_out=None,val_out=None,):
    if 'learn' in mode:
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = simple_genetic_search(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            limit=300,
            report_best= lambda net: net.save(),
        )
    elif 'reproduce' in mode:   # reproduce the best result from the grid search log
        if train_out is None:
            train_out = train
        if val_out is None:
            val_out = val
        ae, _, _ = reproduce(
            curry(nn_task, latplan.model.get(default_parameters["aeclass"]),
                  path,
                  train, train_out, val, val_out), # noise data is used for tuning metric
            default_parameters,
            parameters,
            path,
            report_best= lambda net: net.save(),
        )
        ae.save()
    else:
        ae = latplan.model.load(path)
    return ae

def show_summary(ae,train,test):
    if 'summary' in mode:
        ae.summary()
        ae.report(train, test_data=test, train_data_to=train, test_data_to=test)

################################################################

def puzzle(type='mnist',width=3,height=3,num_examples=6500,N=None,num_actions=None,direct=None,stop_gradient=False,aeclass="ConvolutionalGumbelAE",comment=""):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    import importlib
    p = importlib.import_module('latplan.puzzles.puzzle_{}'.format(type))
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["puzzle",type,width,height]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]
    pres = p.generate(pre_configs,width,height)
    sucs = p.generate(suc_configs,width,height)
    transitions = np.array([pres, sucs])
    states = np.concatenate((transitions[0], transitions[1]), axis=0)
    data = np.swapaxes(transitions,0,1)
    print(data.shape)
    train = data[:int(len(data)*0.9)]
    val   = data[int(len(data)*0.9):int(len(data)*0.95)]
    test  = data[int(len(data)*0.95):]
    ae = run(os.path.join("samples",sae_path), train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae,test,train)
    dump_actions(ae,transitions)
    dump_states(ae,states)

def hanoi(disks=7,towers=4,num_examples=6500,N=None,num_actions=None,direct=None,stop_gradient=False,aeclass="ConvolutionalGumbelAE",comment=""):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    import latplan.puzzles.hanoi as p
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["hanoi",disks,towers]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres']
        suc_configs = data['sucs']
    pres = p.generate(pre_configs,disks,towers)
    sucs = p.generate(suc_configs,disks,towers)
    transitions = np.array([pres, sucs])
    
    states = np.concatenate((transitions[0], transitions[1]), axis=0)
    data = np.swapaxes(transitions,0,1)
    print(data.shape)
    train = data[:int(num_examples*0.9)]
    val   = data[int(num_examples*0.9):int(num_examples*0.95)]
    test  = data[int(num_examples*0.95):]
    print(train.shape, val.shape, test.shape)
    ae = run(os.path.join("samples",sae_path), train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae,test,train)
    dump_actions(ae,transitions)
    dump_states(ae,states)

def lightsout(type='digital',size=4,num_examples=6500,N=None,num_actions=None,direct=None,stop_gradient=False,aeclass="ConvolutionalGumbelAE",comment=""):
    for name, value in locals().items():
        if value is not None:
            parameters[name] = [value]
    default_parameters["aeclass"] = aeclass

    import importlib
    p = importlib.import_module('latplan.puzzles.lightsout_{}'.format(type))
    p.setup()
    path = os.path.join(latplan.__path__[0],"puzzles","-".join(map(str,["lightsout",type,size]))+".npz")
    with np.load(path) as data:
        pre_configs = data['pres'][:num_examples]
        suc_configs = data['sucs'][:num_examples]
    pres = p.generate(pre_configs)
    sucs = p.generate(suc_configs)
    transitions = np.array([pres, sucs])
    states = np.concatenate((transitions[0], transitions[1]), axis=0)
    data = np.swapaxes(transitions,0,1)
    print(data.shape)
    train = data[:int(len(data)*0.9)]
    val   = data[int(len(data)*0.9):int(len(data)*0.95)]
    test  = data[int(len(data)*0.95):]
    ae = run(os.path.join("samples",sae_path), train, val, parameters)
    show_summary(ae, train, test)
    plot_autoencoding_image(ae,test,train)
    dump_actions(ae,transitions)
    dump_states(ae,states)

def main():
    global mode, sae_path
    import sys
    if len(sys.argv) == 1:
        print({ k for k in dir(latplan.model)})
        gs = globals()
        print({ k for k in gs if hasattr(gs[k], '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        mode = sys.argv.pop(0)
        sae_path = "_".join(sys.argv)
        task = sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str
        
        globals()[task](*map(myeval,sys.argv))
    
if __name__ == '__main__':
    try:
        main()
    except:
        import latplan.util.stacktrace
        latplan.util.stacktrace.format()

#!/usr/bin/env python3

"""
I don't remember what it does
"""


import warnings
import config
import numpy as np
from model import GumbelAE, GumbelAE2, Discriminator, default_networks
from model import GumbelAETan

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

lr = 0.001
batch_size = 100
epoch = 100
max_temperature = 2.0
min_temperature = 0.1

from plot import plot_ae

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def prepare(data):
    num = len(data)
    dim = data.shape[1]//2
    print("in prepare: ",data.shape,num,dim)
    pre, suc = data[:,:dim], data[:,dim:]
    
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)

    diff_valid   = suc         - pre
    diff_invalid = suc_invalid - pre
    
    inputs = np.concatenate((diff_valid,diff_invalid),axis=0)
    outputs = np.concatenate((np.ones((num,1)),np.zeros((num,1))),axis=0)
    print("in prepare: ",inputs.shape,outputs.shape)
    io = np.concatenate((inputs,outputs),axis=1)
    random.shuffle(io)

    train_n = int(2*num*0.9)
    train, test = io[:train_n], io[train_n:]
    train_in, train_out = train[:,:dim], train[:,dim:]
    test_in, test_out = test[:,:dim], test[:,dim:]
    print("in prepare: ",train_in.shape, train_out.shape, test_in.shape, test_out.shape)
    
    return train_in, train_out, test_in, test_out

def prepare2(data):
    "valid data diff only"
    num = len(data)
    dim = data.shape[1]//2
    print("in prepare: ",data.shape,num,dim)
    pre, suc = data[:,:dim], data[:,dim:]
    
    diff_valid   = suc         - pre
    
    inputs = diff_valid
    outputs = np.ones((num,1))
    print("in prepare: ",inputs.shape,outputs.shape)
    io = np.concatenate((inputs,outputs),axis=1)
    random.shuffle(io)

    train_n = int(num*0.9)
    train, test = io[:train_n], io[train_n:]
    train_in, train_out = train[:,:dim], train[:,dim:]
    test_in, test_out = test[:,:dim], test[:,dim:]
    print("in prepare: ",train_in.shape, train_out.shape, test_in.shape, test_out.shape)
    
    return train_in, train_out, test_in, test_out

def prepare3(data):
    "valid data only"
    num = len(data)
    dim = data.shape[1]//2
    print("in prepare: ",data.shape,num,dim)
    
    inputs = data
    outputs = np.ones((num,1))
    print("in prepare: ",inputs.shape,outputs.shape)
    io = np.concatenate((inputs,outputs),axis=1)
    random.shuffle(io)

    train_n = int(num*0.9)
    train, test = io[:train_n], io[train_n:]
    train_in, train_out = train[:,:2*dim], train[:,2*dim:]
    test_in, test_out = test[:,:2*dim], test[:,2*dim:]
    print("in prepare: ",train_in.shape, train_out.shape, test_in.shape, test_out.shape)
    
    return train_in, train_out, test_in, test_out


def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return math.log(max/min) / epoch

def grid_search(path, train_in, train_out, test_in, test_out):
    global lr, batch_size, epoch, max_temperature, min_temperature
    names      = ['layer','dropout','N']
    parameters = [[1000],[0.4],[25]]
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    try:
        import itertools
        import tensorflow as tf
        for params in itertools.product(*parameters):
            params_dict = { k:v for k,v in zip(names,params) }
            print("Testing model with parameters={}".format(params_dict))
            ae = GumbelAE(path,params_dict)
            finished = False
            while not finished:
                print("batch size {}".format(batch_size))
                try:
                    ae.train(train_in,
                             test_data=test_in,
                             train_data_to=train_out,
                             test_data_to=test_out,
                             # 
                             epoch=epoch,
                             anneal_rate=anneal_rate(epoch,min_temperature,max_temperature),
                             max_temperature=max_temperature,
                             min_temperature=min_temperature,
                             lr=lr,
                             batch_size=batch_size,)
                    finished = True
                except tf.errors.ResourceExhaustedError as e:
                    print(e)
                    batch_size = batch_size // 2

            if isinstance(ae.loss,list):
                error = ae.net.evaluate(
                    test_in,[test_out,test_out],batch_size=batch_size,)[0]
            else:
                error = ae.net.evaluate(
                    test_in,test_out,batch_size=batch_size,)
            results.append((error,)+params)
            print("Evaluation result for {} : error = {}".format(params_dict,error))
            print("Current results:\n{}".format(np.array(results)),flush=True)
            if error < best_error:
                print("Found a better parameter {}: error:{} old-best:{}".format(
                    params_dict,error,best_error))
                best_params = params_dict
                best_error = error
                best_ae = ae
        print("Best parameter {}: error:{}".format(best_params,best_error))
    finally:
        print(results)
    best_ae.save()
    return best_ae,best_params,best_error


if __name__ == '__main__':
    import numpy.random as random
    from trace import trace

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_ae = "{}_ae/".format(directory)
    
    data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
    train_in, train_out, test_in, test_out = prepare3(data)
    
    train = True
    if train:
        ae, _, _ = grid_search(directory_ae, train_in, train_in, test_in, test_in)
    else:
        ae = GumbelAE(directory_ae).load()

    show_n = 10
    print("round(_y-y)")
    for y,_y in zip(ae.autoencode(test_in)[:show_n],
                    test_in[:show_n]):
        print(np.array(np.round(_y-y),dtype=np.int))

    # test if the learned action is correct
        
    # actions_valid = np.loadtxt("{}/all_actions.csv".format(directory),dtype=int)
    # ae = default_networks['fc2'](directory).load()
    # N = ae.parameters["N"]
    # 
    # actions_invalid = np.random.randint(0,2,(10000,2*N))
    # print(actions_valid.shape,actions_valid.dtype,actions_invalid.shape,actions_invalid.dtype)
    # 
    # ai = actions_invalid.view([('', actions_invalid.dtype)] * 2*N)
    # av = actions_valid.view  ([('', actions_valid.dtype)]   * 2*N)
    # actions_invalid = np.setdiff1d(ai, av).view(actions_invalid.dtype).reshape((-1, 2*N))
    # 
    # print("invalid",actions_invalid.shape)

    # ae.report(actions_valid,  train_data_to=np.ones((len(actions_valid),)))
    # ae.report(actions_invalid,train_data_to=np.zeros((len(actions_invalid),)))

"""

* Summary:

Tries to compress the state pairs (2*N) to an even smaller latent space

failed

"""

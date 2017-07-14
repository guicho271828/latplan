#!/usr/bin/env python3
import warnings
import config
import numpy as np
from model import GumbelAE, Discriminator, default_networks

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

def select(data,num):
    return data[random.randint(0,data.shape[0],num)]

def prepare(data):
    num = len(data)
    dim = data.shape[1]//2
    print(data.shape,num,dim)
    pre, suc = data[:,:dim], data[:,dim:]
    
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    data_invalid = np.concatenate((pre,suc_invalid),axis=1)

    ai = data_invalid.view([('', data_invalid.dtype)] * 2*dim)
    av = data.view        ([('', data.dtype)]         * 2*dim)
    data_invalid = np.setdiff1d(ai, av).view(data_invalid.dtype).reshape((-1, 2*dim))
    
    inputs = np.concatenate((data,data_invalid),axis=0)
    outputs = np.concatenate((np.ones((num,1)),np.zeros((len(data_invalid),1))),axis=0)
    print(inputs.shape,outputs.shape)
    io = np.concatenate((inputs,outputs),axis=1)
    random.shuffle(io)

    train_n = int(2*num*0.9)
    train, test = io[:train_n], io[train_n:]
    train_in, train_out = train[:,:dim*2], train[:,dim*2:]
    test_in, test_out = test[:,:dim*2], test[:,dim*2:]
    
    return train_in, train_out, test_in, test_out
    

def curry(fn,*args1,**kwargs1):
    return lambda *args,**kwargs: fn(*args1,*args,**{**kwargs1,**kwargs})

def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return math.log(max/min) / epoch

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
}
parameters = {}

def learn_model(path,train_in,train_out,test_in,test_out,network,params_dict={}):
    discriminator = network(path)
    training_parameters = default_parameters.copy()
    for key, _ in training_parameters.items():
        if key in params_dict:
            training_parameters[key] = params_dict[key]
    discriminator.train(train_in,
                        test_data=test_in,
                        train_data_to=train_out,
                        test_data_to=test_out,
                        anneal_rate=anneal_rate(training_parameters['full_epoch'],
                                                training_parameters['min_temperature'],
                                                training_parameters['max_temperature']),
                        report=False,
                        **training_parameters,)
    return discriminator

def grid_search(path, train_in, train_out, test_in, test_out):
    # perform random trials on possible combinations
    network = Discriminator
    best_error = float('inf')
    best_params = None
    best_ae     = None
    results = []
    print("Network: {}".format(network))
    try:
        import itertools
        names  = [ k for k, _ in parameters.items()]
        values = [ v for _, v in parameters.items()]
        all_params = list(itertools.product(*values))
        random.shuffle(all_params)
        [ print(r) for r in all_params]
        for i,params in enumerate(all_params):
            config.reload_session()
            params_dict = { k:v for k,v in zip(names,params) }
            print("{}/{} Testing model with parameters=\n{}".format(i, len(all_params), params_dict))
            ae = learn_model(path, train_in,train_out,test_in,test_out,
                             network=curry(network, parameters=params_dict),
                             params_dict=params_dict)
            error = ae.net.evaluate(test_in,test_out,batch_size=100,verbose=0)
            results.append({'error':error, **params_dict})
            print("Evaluation result for:\n{}\nerror = {}".format(params_dict,error))
            print("Current results:")
            results.sort(key=lambda result: result['error'])
            [ print(r) for r in results]
            if error < best_error:
                print("Found a better parameter:\n{}\nerror:{} old-best:{}".format(
                    params_dict,error,best_error))
                del best_ae
                best_params = params_dict
                best_error = error
                best_ae = ae
            else:
                del ae
        print("Best parameter:\n{}\nerror: {}".format(best_params,best_error))
    finally:
        print(results)
    best_ae.save()
    with open(best_ae.local("grid_search.log"), 'a') as f:
        import json
        f.write("\n")
        json.dump(results, f)
    return best_ae,best_params,best_error


if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_ad = "{}/_ad/".format(directory)
    
    data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
    train_in, train_out, test_in, test_out = prepare(data)
    
    global parameters
    parameters = {
        'layer'      :[300],# [400,4000],
        'dropout'    :[0.1], #[0.1,0.4],
        'batch_size' :[1000],
        'full_epoch' :[1000],
        'activation' :['tanh'],
        # quick eval
        'epoch'      :[50],
        'lr'         :[0.0001],
    }
    
    try:
        discriminator = Discriminator(directory_ad).load()
    except FileNotFoundError:
        discriminator,_,_ = grid_search(directory_ad, train_in, train_out, test_in, test_out)

    show_n = 30
    
    for y,_y in zip(discriminator.discriminate(test_in)[:show_n],
                    test_out[:show_n]):
        print(y,_y)

    # test if the learned action is correct

    actions_valid = np.loadtxt("{}/all_actions.csv".format(directory),dtype=int)
    
    from latplan.util import get_ae_type
    ae = default_networks[get_ae_type(directory)](directory).load()
    N = ae.parameters["N"]
    print("valid",actions_valid.shape)
    discriminator.report(actions_valid,  train_data_to=np.ones((len(actions_valid),)))

    # invalid actions generated from random bits
    actions_invalid = np.random.randint(0,2,(len(actions_valid),2*N))
    ai = actions_invalid.view([('', actions_invalid.dtype)] * 2*N)
    av = actions_valid.view  ([('', actions_valid.dtype)]   * 2*N)
    actions_invalid = np.setdiff1d(ai, av).view(actions_invalid.dtype).reshape((-1, 2*N))
    print("invalid",actions_invalid.shape)
    discriminator.report(actions_invalid,train_data_to=np.zeros((len(actions_invalid),)))

    # invalid actions generated from swapping successors; predecessors/successors are both correct states
    pre, suc = actions_valid[:,:N], actions_valid[:,N:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    actions_invalid2 = np.concatenate((pre,suc_invalid),axis=1)
    ai = actions_invalid2.view([('', actions_invalid2.dtype)] * 2*N)
    av = actions_valid.view  ([('', actions_valid.dtype)]   * 2*N)
    actions_invalid2 = np.setdiff1d(ai, av).view(actions_invalid2.dtype).reshape((-1, 2*N))
    print("invalid2",actions_invalid2.shape)
    discriminator.report(actions_invalid2,train_data_to=np.zeros((len(actions_invalid2),)))

    actions_invalid3 = actions_invalid.copy()
    actions_invalid3[:,:N] = actions_valid[:len(actions_invalid),:N]
    print("invalid3",actions_invalid3.shape)
    discriminator.report(actions_invalid3,train_data_to=np.zeros((len(actions_invalid3),)))
    
    actions_invalid4 = actions_invalid.copy()
    actions_invalid4[:,N:] = actions_valid[:len(actions_invalid),N:]
    print("invalid4",actions_invalid4.shape)
    discriminator.report(actions_invalid4,train_data_to=np.zeros((len(actions_invalid4),)))
    
    
"""

* Summary:
Input: a subset of valid action pairs.

* Training:
From the valid action pairs, pseudo-invalid action pairs are generated by randomly swapping the successor states.
(graph sparsity assumption)

Oracle is trained to classify valid and pseudo-invalid action pairs. output of
the network is a single value indicating valid (1) and invalid (0).

The valid and the pseudo-invalid pairs are concatenated, randomly reordered and
divided into training samples and validation samples (for checking if
it is not overfitting. This does not by itself show the correctness of the
learned model)

* Evaluation:

Dataset: mnist_puzzle33_fc2

The result is validated on the entire valid action set (967680) and 2 sets of invalid action set.
Invalid action set (1) is created by removing the valid actions from a set of 10000 randomly generated bit vectors.
Invalid action set (2) is created by removing the valid actions from a set generated by swapping the successors of valid actions.
This guarantees all "states" are correct; thus the NN is not detecting the "unrealistic" states, and purely looking at the transitions.
Invalid action set (3): from invalid states to valid states.
Invalid action set (4): from valid states to invalid states.

type-1 error for the entire valid action set (967680 actions):
Mean Absolute Error: 

type-2 error for the invalid action set (1):
Mean Absolute Error: 

type-2 error for the invalid action set (2):
Mean Absolute Error: 

type-2 error for the invalid action set (3):
Mean Absolute Error: 

type-2 error for the invalid action set (4):
Mean Absolute Error: 


"""

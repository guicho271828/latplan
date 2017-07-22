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
# State discriminator.
# 
# I made this discriminator for pruning some state transitions while checking
# the entire (2^98) transitions and make them compact.

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
    directory_sd = "{}/_sd/".format(directory)
    
    data_valid = np.loadtxt("{}/states.csv".format(directory),dtype=np.int8)
    print(data_valid.shape)
    batch = data_valid.shape[0]
    N = data_valid.shape[1]
    data_invalid = np.random.randint(0,2,(batch*10,N),dtype=np.int8)
    print(data_valid.shape,data_invalid.shape)
    ai = data_invalid.view([('', data_invalid.dtype)] * N)
    av = data_valid.view  ([('', data_valid.dtype)]   * N)
    data_invalid = np.setdiff1d(ai, av).view(data_valid.dtype).reshape((-1, N))

    out_valid   = np.ones ((len(data_valid),1))
    out_invalid = np.zeros((len(data_invalid),1))
    data_out = np.concatenate((out_valid, out_invalid),axis=0)
    data_in  = np.concatenate((data_valid, data_invalid),axis=0)

    train_in  = data_in [:int(0.9*len(data_out))]
    train_out = data_out[:int(0.9*len(data_out))]
    test_in   = data_in [int(0.9*len(data_out)):]
    test_out  = data_out[int(0.9*len(data_out)):]

    global parameters
    parameters = {
        'layer'      :[300],# [400,4000],
        'dropout'    :[0.1], #[0.1,0.4],
        'num_layers' :[2],
        'batch_size' :[1000],
        'full_epoch' :[1000],
        'activation' :['tanh'],
        # quick eval
        'epoch'      :[200],
        'lr'         :[0.0001],
    }

    try:
        discriminator = Discriminator(directory_sd).load()
    except:
        discriminator,_,_ = grid_search(directory_sd, train_in, train_out, test_in, test_out)
        
    print("index, discrimination, action")
    show_n = 30

    for y,_y in zip(discriminator.discriminate(test_in)[:show_n],
                    test_out[:show_n]):
        print(y,_y)

    # test if the learned action is correct

    states_valid = np.loadtxt("{}/all_states.csv".format(directory),dtype=int)
    ae = default_networks['fc'](directory).load()
    N = ae.parameters["N"]
    print("valid",states_valid.shape)

    # invalid states generated from random bits
    states_invalid = np.random.randint(0,2,(len(states_valid),N))
    ai = states_invalid.view([('', states_invalid.dtype)] * N)
    av = states_valid.view  ([('', states_valid.dtype)]   * N)
    states_invalid = np.setdiff1d(ai, av).view(states_invalid.dtype).reshape((-1, N))
    print("invalid",states_invalid.shape)

    type1_error = np.sum(1- discriminator.discriminate(states_valid,batch_size=1000).round())
    print("type1 error:",type1_error,"/",len(states_valid),
          "Error ratio:", type1_error/len(states_valid) * 100, "%")
    type2_error = np.sum(discriminator.discriminate(states_invalid,batch_size=1000).round())
    print("type2 error:",type2_error,"/",len(states_invalid),
          "Error ratio:", type2_error/len(states_invalid) * 100, "%")
    
"""

This is a wrong attempt; the better the SAE compression, the less the "invalid" states.

* Summary:
Input: a subset of valid states and random bitstrings (invalid states)
Output: a function that returns 0/1 for a state

* Training:


* Network:


* Evaluation:

Dataset and network: mnist_puzzle33_fc2

type-1 error for the entire valid states (967680 states):
MAE: 0.09277774112190677

type-2 error for the invalid states:
MAE: 0.03727853568254886

"""

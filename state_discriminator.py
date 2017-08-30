#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import Discriminator, default_networks
from latplan.util        import curry, prepare_binary_classification_data
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################
# State discriminator.
# 
# I made this discriminator for pruning some state transitions while checking
# the entire (2^98) transitions and make them compact.

def prepare(data_valid):
    print(data_valid.shape)
    batch = data_valid.shape[0]
    N = data_valid.shape[1]
    data_invalid = np.random.randint(0,2,(batch,N),dtype=np.int8)
    print(data_valid.shape,data_invalid.shape)
    ai = data_invalid.view([('', data_invalid.dtype)] * N)
    av = data_valid.view  ([('', data_valid.dtype)]   * N)
    data_invalid = np.setdiff1d(ai, av).view(data_valid.dtype).reshape((-1, N))

    return prepare_binary_classification_data(data_valid, data_invalid)

# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
    'M'               : 2,
}
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

if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_sd = "{}/_sd/".format(directory)
    
    data_valid = np.loadtxt("{}/states.csv".format(directory),dtype=np.int8)
    train_in, train_out, test_in, test_out = prepare(data_valid)

    try:
        discriminator = Discriminator(directory_sd).load()
    except (FileNotFoundError, ValueError):
        discriminator,_,_ = grid_search(curry(nn_task, Discriminator, directory_sd,
                                              train_in, train_out, test_in, test_out,),
                                        default_parameters,
                                        parameters)
        discriminator.save()
        
    print("index, discrimination, action")
    show_n = 30

    for y,_y in zip(discriminator.discriminate(test_in)[:show_n],
                    test_out[:show_n]):
        print(y,_y)

    # test if the learned action is correct

    states_valid = np.loadtxt("{}/all_states.csv".format(directory),dtype=int)
    from latplan.util import get_ae_type
    ae = default_networks[get_ae_type(directory)](directory).load()
    N = ae.parameters["N"]
    print("valid",states_valid.shape)

    # invalid states generated from random bits
    states_invalid = np.random.randint(0,2,(len(states_valid),N))
    ai = states_invalid.view([('', states_invalid.dtype)] * N)
    av = states_valid.view  ([('', states_valid.dtype)]   * N)
    states_invalid = np.setdiff1d(ai, av).view(states_invalid.dtype).reshape((-1, N))
    print("invalid",states_invalid.shape)

    from latplan.util.plot import plot_grid

    type1_d = discriminator.discriminate(states_valid,batch_size=1000).round()
    type1_error = np.sum(1- type1_d)
    print("type1 error:",type1_error,"/",len(states_valid),
          "Error ratio:", type1_error/len(states_valid) * 100, "%")
    plot_grid(ae.decode_binary(states_valid[np.where(type1_d < 0.1)[0]])[:20],
              path=discriminator.local("type1_error.png"))

    type2_d = discriminator.discriminate(states_invalid,batch_size=1000).round()
    type2_error = np.sum(type2_d)
    print("type2 error:",type2_error,"/",len(states_invalid),
          "Error ratio:", type2_error/len(states_invalid) * 100, "%")
    plot_grid(ae.decode_binary(states_invalid[np.where(type2_d > 0.9)[0]])[:20],
              path=discriminator.local("type2_error.png"))

    
    
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

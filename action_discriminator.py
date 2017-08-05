#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import PUDiscriminator, default_networks
from latplan.util        import curry, set_difference, prepare_binary_classification_data
from latplan.util.tuning import grid_search, nn_task

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

ae = None

def load_ae(directory):
    global ae
    if ae is None:
        from latplan.util import get_ae_type
        ae = default_networks[get_ae_type(directory)](directory).load()
    return ae

inflation = 1

def generate_nop(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    pre = np.concatenate((pre, suc), axis=0)
    data_invalid = np.concatenate((pre,pre),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def permute_suc(data):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    data_invalid = np.concatenate((pre,suc_invalid),axis=1)
    data_invalid = set_difference(data_invalid, data)
    return data_invalid

def generate_random_action(data, sae):
    dim = data.shape[1]//2
    pre, suc = data[:,:dim], data[:,dim:]
    from state_discriminator3 import generate_random
    pre = np.concatenate((pre, suc), axis=0)
    suc = np.concatenate((generate_random(pre, sae),
                          generate_random(pre, sae)), axis=0)[:len(pre)]
    actions_invalid = np.concatenate((pre, suc), axis=1)
    actions_invalid = set_difference(actions_invalid, data)
    return actions_invalid
    
def prepare(data):
    data_invalid = np.concatenate(
        tuple([generate_nop(data),
               *[ permute_suc(data) for i in range(inflation) ],
               *[ generate_random_action(data, ae) for i in range(inflation) ]
        ]), axis=0)

    data_valid   = np.repeat(data, len(data_invalid)//len(data), axis=0)

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
    'num_layers' :[2],
    'layer'      :[300],# [400,4000],
    'dropout'    :[0.1], #[0.1,0.4],
    'batch_size' :[1000],
    'full_epoch' :[1000],
    'activation' :['tanh'],
    # quick eval
    'epoch'      :[300],
    'lr'         :[0.0001],
}


if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) != 3:
        sys.exit("{} [directory] [mode]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_ad = "{}/_ad/".format(directory)
    mode = sys.argv[2]

    try:
        if 'learn' in mode:
            raise Exception('learn')
        discriminator = PUDiscriminator(directory_ad).load()
    except:
        data = np.loadtxt("{}/actions.csv".format(directory),dtype=np.int8)
        load_ae(directory)
        train_in, train_out, test_in, test_out = prepare(data)

        discriminator,_,_ = grid_search(curry(nn_task, PUDiscriminator, directory_ad,
                                              train_in, train_out, test_in, test_out,),
                                        default_parameters,
                                        parameters)
        discriminator.save()
    
    # test if the learned action is correct

    # actions_valid = np.loadtxt("{}/actions.csv".format(directory),dtype=int)
    actions_valid = np.loadtxt("{}/all_actions.csv".format(directory),dtype=np.int8)
    random.shuffle(actions_valid)
    actions_valid = actions_valid[:100000]
    
    N = actions_valid.shape[1] // 2
    print("valid",actions_valid.shape)
    discriminator.report(actions_valid,  train_data_to=np.ones((len(actions_valid),)))
    print("type1 error: ",np.mean(1-np.round(discriminator.discriminate(actions_valid,batch_size=1000)))*100, "%")

    c = 0
    def type2(actions_invalid, message):
        global c
        c += 1
        actions_invalid = set_difference(actions_invalid, actions_valid)
        print("invalid",c,actions_invalid.shape, "---", message)
        discriminator.report(actions_invalid,train_data_to=np.zeros((len(actions_invalid),)))
        print("type2 error:",np.mean(np.round(discriminator.discriminate(actions_invalid,batch_size=1000))) * 100, "%")

    type2(np.random.randint(0,2,(len(actions_valid),2*N),dtype=np.int8),
          "invalid actions generated from random bits (both pre and suc)")
        
    load_ae(directory)
    type2(generate_random_action(actions_valid, ae),
          "sucessors are random reconstructable states")
    
    pre, suc = actions_valid[:,:N], actions_valid[:,N:]
    suc_invalid = np.copy(suc)
    random.shuffle(suc_invalid)
    type2(np.concatenate((pre,suc_invalid),axis=1),
          "generated by swapping successors; pre/suc are correct states")

    if 'check' in mode:
        import latplan.puzzles.puzzle_mnist as p
        p.setup()
        import latplan.puzzles.model.puzzle as m
        count = 0
        batch = 10000
        for i in range(len(pre)//batch):
            pre_images = ae.decode_binary(pre        [batch*i:batch*(i+1)],batch_size=1000)
            suc_images = ae.decode_binary(suc_invalid[batch*i:batch*(i+1)],batch_size=1000)
            m.validate_transitions([pre_images, suc_images], 3,3)
            count += np.count_nonzero(validation)
        print(count,"valid actions in invalid2")
    
    pre2 = np.loadtxt("{}/all_states.csv".format(directory),dtype=np.int8)
    suc2 = np.copy(pre2)
    random.shuffle(suc2)
    type2(np.concatenate((pre2,suc2),axis=1),
          "generated by shuffling all valid states; pre/suc are correct but possibly unknown states")
    
    if 'check' in mode:
        count = 0
        batch = 10000
        for i in range(len(pre)//batch):
            pre_images = ae.decode_binary(pre2[batch*i:batch*(i+1)],batch_size=1000)
            suc_images = ae.decode_binary(suc2[batch*i:batch*(i+1)],batch_size=1000)
            m.validate_transitions([pre_images, suc_images], 3,3)
            count += np.count_nonzero(validation)
        print(count,"valid actions in invalid2")

    type2(np.concatenate((pre2, pre2), axis=1),
          "invalid actions generated from nop")

    type2(np.concatenate((np.random.randint(0,2,(len(actions_valid),N),dtype=np.int8), suc), axis=1),
          "pre are generated by random bits, suc are correct states")

    type2(np.concatenate((pre, np.random.randint(0,2,(len(actions_valid),N),dtype=np.int8)), axis=1),
          "suc are generated by random bits, pre are correct states")
    
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

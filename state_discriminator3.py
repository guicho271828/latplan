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

# negative examples (random bitstrings) are pre-filtered using SAE reconstruction

threshold = 0.01
rate_threshold = 0.99
max_repeat = 50

def bce(x,y,axis):
    return - (x * np.log(y+1e-5) + \
              (1-x) * np.log(1-y+1e-5)).mean(axis=axis)

def prepare(data_valid, ae):
    print(data_valid.shape)
    batch = data_valid.shape[0]
    N = data_valid.shape[1]

    data_invalid = np.random.randint(0,2,(batch,N),dtype=np.int8)

    loss = 1000000000
    for i in range(max_repeat):
        images           = ae.decode_binary(data_invalid,batch_size=2000)
        data_invalid_rec = ae.encode_binary(images,batch_size=2000)

        prev_loss = loss
        loss    = bce(data_invalid,data_invalid_rec,(0,1,))
        print(loss, loss / prev_loss)
        data_invalid = data_invalid_rec
        if loss / prev_loss > rate_threshold:
            break
        if loss < threshold:
            break

    data_invalid = data_invalid.round().astype(np.int8)
    from latplan.util import set_difference
    data_invalid = set_difference(data_invalid, data_valid)
    print(batch, " -> ", len(data_invalid), "invalid examples")
    train_in, train_out, test_in, test_out = prepare_binary_classification_data(data_valid, data_invalid)
    return train_in, train_out, test_in, test_out, data_invalid


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


if __name__ == '__main__':
    import numpy.random as random

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory] [mode=learn]".format(sys.argv[0]))

    directory    = sys.argv[1]
    mode         = sys.argv[2]
    directory_sd = "{}/_sd3/".format(directory)
    import subprocess
    subprocess.call(["mkdir","-p",directory_sd])
    
    from latplan.util import get_ae_type
    sae = default_networks[get_ae_type(directory)](directory).load()
 
    if 'learn' in mode:
        data_valid = np.loadtxt(sae.local("states.csv"),dtype=np.int8)

        train_in, train_out, test_in, test_out, data_invalid = prepare(data_valid,sae)

        sae.plot_autodecode(data_invalid[:8], "_sd3/fake_samples.png")

        train_image, test_image = sae.decode_binary(train_in), sae.decode_binary(test_in)
        cae,_,_ = grid_search(curry(nn_task, default_networks['SimpleCAE'],
                                    sae.local("_cae"),
                                    train_image, train_image, test_image, test_image),
                              default_parameters,
                              {
                                  'num_layers' :[2],
                                  'layer'      :[500],
                                  'clayer'     :[16],
                                  'dropout'    :[0.4],
                                  'batch_size' :[4000],
                                  'full_epoch' :[1000],
                                  'activation' :['relu'],
                                  'epoch'      :[300],
                                  'lr'         :[0.001],
                              })

        train_in2, test_in2 = cae.encode(train_image), cae.encode(test_image)
        
        discriminator,_,_ = grid_search(curry(nn_task, Discriminator, directory_sd,
                                              train_in2, train_out, test_in2, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[2],
                                            'layer'      :[500],
                                            'clayer'     :[16],
                                            'dropout'    :[0.4],
                                            'batch_size' :[4000],
                                            'full_epoch' :[1000],
                                            'activation' :['relu'],
                                            'epoch'      :[300],
                                            'lr'         :[0.001],
                                        })
        
    else:
        cae           = default_networks['SimpleCAE'    ](sae.local("_cae")).load()
        discriminator = default_networks['Discriminator'](directory_sd).load()

    def combined_discriminate(data,**kwargs):
        images = sae.decode_binary(data,**kwargs)
        data2  = cae.encode(images,**kwargs)
        return discriminator.discriminate(data2,**kwargs)

    # test if the learned action is correct

    states_valid = np.loadtxt("{}/all_states.csv".format(directory),dtype=np.int8)
    print("valid",states_valid.shape)

    from latplan.util.plot import plot_grid

    type1_d = combined_discriminate(states_valid,batch_size=1000).round()
    type1_error = np.sum(1- type1_d)
    print("type1 error:",type1_error,"/",len(states_valid),
          "Error ratio:", type1_error/len(states_valid) * 100, "%")
    plot_grid(sae.decode_binary(states_valid[np.where(type1_d < 0.1)[0]])[:120],
              w=20,
              path=discriminator.local("type1_error.png"))

    _,_,_,_, states_invalid = prepare(states_valid,sae)
    
    type2_d = combined_discriminate(states_invalid,batch_size=1000).round()
    type2_error = np.sum(type2_d)
    print("type2 error:",type2_error,"/",len(states_invalid),
          "Error ratio:", type2_error/len(states_invalid) * 100, "%")
    plot_grid(sae.decode_binary(states_invalid[np.where(type2_d > 0.9)[0]])[:120],
              w=20,
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

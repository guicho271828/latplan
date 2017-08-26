#!/usr/bin/env python3
import warnings
import config
import numpy as np
from latplan.model import PUDiscriminator, default_networks, combined_discriminate, combined_discriminate2
from latplan.util        import curry, prepare_binary_classification_data, set_difference, bce
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

inflation = 1

def generate_random(data,sae):
    threshold = 0.01
    rate_threshold = 0.99
    max_repeat = 50

    def regenerate(sae, data):
        images           = sae.decode_binary(data,batch_size=2000)
        data_invalid_rec = sae.encode_binary(images,batch_size=2000)
        return data_invalid_rec

    def regenerate_many(sae, data):
        loss = 1000000000
        for i in range(max_repeat):
            data_rec = regenerate(sae, data)
            prev_loss = loss
            loss    = bce(data,data_rec)
            print(loss, loss / prev_loss)
            data = data_rec
            if loss / prev_loss > rate_threshold:
                print("improvement saturated: loss / prev_loss = ", loss / prev_loss, ">", rate_threshold)
                break
            # if loss < threshold:
            #     print("good amount of loss:", loss, "<", threshold)
            #     break
        return data.round().astype(np.int8)
    
    def prune_unreconstructable(sae,data):
        rec = regenerate(sae,data)
        loss = bce(data,rec,(1,))
        return data[np.where(loss < threshold)[0]]
    
    batch = data.shape[0]
    N     = data.shape[1]
    data_invalid = np.random.randint(0,2,(batch,N),dtype=np.int8)
    data_invalid = regenerate_many(sae, data_invalid)
    data_invalid = prune_unreconstructable(sae, data_invalid)
    from latplan.util import set_difference
    data_invalid = set_difference(data_invalid.round(), data.round())
    return data_invalid

def prepare(data_valid, sae):
    data_invalid = generate_random(data_valid, sae)

    batch = data_valid.shape[0]
    for i in range(inflation-1):
        data_invalid = np.concatenate((data_invalid,
                                       set_difference(generate(),data_invalid))
                                      , axis=0)
        print(batch, "->", len(data_invalid), "invalid examples")
    if inflation != 1:
        real_inflation = len(data_invalid)//batch
        data_invalid = data_invalid[:batch*real_inflation]
        data_valid   = np.repeat(data_valid, real_inflation, axis=0)

    train_in, train_out, test_in, test_out = prepare_binary_classification_data(data_valid, data_invalid)
    return train_in, train_out, test_in, test_out, data_valid, data_invalid


# default values
default_parameters = {
    'lr'              : 0.0001,
    'batch_size'      : 2000,
    'full_epoch'      : 1000,
    'epoch'           : 1000,
    'max_temperature' : 2.0,
    'min_temperature' : 0.1,
    'M'               : 2,
    'min_grad'        : 0.0,
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
    aetype = get_ae_type(directory)
    sae = default_networks[aetype](directory).load()
 
    if 'learn' in mode:
        data_valid = np.loadtxt(sae.local("states.csv"),dtype=np.int8)

        train_in, train_out, test_in, test_out, data_valid, data_invalid = prepare(data_valid,sae)

        sae.plot_autodecode(data_invalid[:8], "_sd3/fake_samples.png")

        train_image, test_image = sae.decode_binary(train_in), sae.decode_binary(test_in)
        
        if 'conv' in aetype:
            train_in2, test_in2 = sae.get_features(train_image), sae.get_features(test_image)
        else:
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
                                      'epoch'      :[30],
                                      'lr'         :[0.001],
                                  })

            cae.save()
            train_in2, test_in2 = cae.encode(train_image), cae.encode(test_image)
        
        discriminator,_,_ = grid_search(curry(nn_task, PUDiscriminator, directory_sd,
                                              train_in2, train_out, test_in2, test_out,),
                                        default_parameters,
                                        {
                                            'num_layers' :[1],
                                            'layer'      :[50],
                                            'clayer'     :[16],
                                            'dropout'    :[0.8],
                                            'batch_size' :[1000],
                                            'full_epoch' :[1000],
                                            'activation' :['relu'],
                                            'epoch'      :[3000],
                                            'lr'         :[0.0001],
                                        })
        discriminator.save()
        
    else:
        if 'conv' not in aetype:
            cae           = default_networks['SimpleCAE'    ](sae.local("_cae")).load()
        discriminator = default_networks['PUDiscriminator'](directory_sd).load()

    # test if the learned action is correct

    states_valid = np.loadtxt("{}/all_states.csv".format(directory),dtype=np.int8)
    print("valid",states_valid.shape)

    from latplan.util.plot import plot_grid

    if 'conv' in aetype:
        type1_d = combined_discriminate2(states_valid,sae,discriminator,batch_size=1000).round()
    else:
        type1_d = combined_discriminate(states_valid,sae,cae,discriminator,batch_size=1000).round()
    type1_error = np.sum(1- type1_d)
    print("type1 error:",type1_error,"/",len(states_valid),
          "Error ratio:", type1_error/len(states_valid) * 100, "%")

    type1_error_images = sae.decode_binary(states_valid[np.where(type1_d < 0.1)[0]])[:120]
    if len(type1_error_images) == 0:
        print("We observed ZERO type1-error! Hooray!")
    else:
        plot_grid(type1_error_images,
                  w=20,
                  path=discriminator.local("type1_error.png"))
    
    inflation = 1
    _,_,_,_, _, states_invalid = prepare(states_valid,sae)

    if 'check' in mode:
        import latplan.puzzles.puzzle_mnist as p
        p.setup()
        # p.validate_states(sae.decode_binary(states_valid))
        is_invalid = p.validate_states(sae.decode_binary(states_invalid))
        states_invalid = states_invalid[np.logical_not(is_invalid)]

        plot_grid(sae.decode_binary(states_invalid)[:120],
                  w=20,
                  path=discriminator.local("surely_invalid_states.png"))
    
    if 'conv' in aetype:
        type2_d = combined_discriminate2(states_invalid,sae,discriminator,batch_size=1000).round()
    else:
        type2_d = combined_discriminate(states_invalid,sae,cae,discriminator,batch_size=1000).round()
    type2_error = np.sum(type2_d)
    print("type2 error:",type2_error,"/",len(states_invalid),
          "Error ratio:", type2_error/len(states_invalid) * 100, "%")

    type2_error_images = sae.decode_binary(states_invalid[np.where(type2_d > 0.9)[0]])[:120]
    if len(type2_error_images) == 0:
        print("We observed ZERO type2-error! Hooray!")
    else:
        plot_grid(type2_error_images,
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

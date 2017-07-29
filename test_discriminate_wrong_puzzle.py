#!/usr/bin/env python3
import warnings
import config
import numpy as np
import latplan
from latplan.model import Discriminator, ConvolutionalDiscriminator, default_networks
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

def images(p,num=5000):
    p.setup()
    import latplan.puzzles.model.puzzle
    latplan.puzzles.model.puzzle.load(3,3,True)
    configs = p.generate_configs(9)
    configs = np.array([ c for c in configs ])
    random.shuffle(configs)
    train_c = configs[:num]
    return p.states(3,3,train_c)

def prepare():
    real = images(latplan.puzzles.puzzle_mnist)
    fake = images(latplan.puzzles.puzzle_wrong)
    prepare_binary_classification_data(real,fake)

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
    'layer'      :[500],
    'clayer'     :[16],
    'dropout'    :[0.4],
    'batch_size' :[4000],
    'full_epoch' :[1000],
    'activation' :['relu'],
    'epoch'      :[100],
    'lr'         :[0.001],
}

if __name__ == '__main__':
    import numpy.random as random

    train_in, train_out, test_in, test_out = prepare()

    ae = default_networks['SimpleCAE']("samples/wrong_detector_cae").load()
    
    # ae,_,_ = grid_search(curry(nn_task, default_networks['SimpleCAE'], "samples/wrong_detector_cae",
    #                            train_in, train_in, test_in, test_in,),
    #                      default_parameters,
    #                      parameters)

    recons = ae.autoencode(test_in[:6])
    images = []
    for seq in zip(test_in[:6], recons):
        images.extend(seq)
    from latplan.util.plot import plot_grid
    plot_grid(images, w=8, path=ae.local("reconstruction.png"))

    train_in2 = ae.encode(train_in)
    test_in2 = ae.encode(test_in)
    
    discriminator,_,_ = grid_search(curry(nn_task, Discriminator, "samples/wrong_detector_classify",
                                          train_in2, train_out, test_in2, test_out,),
                                    default_parameters,
                                    parameters)
        
"""

Try to detect a wrong 8puzzle and a correct 8puzzle

"""

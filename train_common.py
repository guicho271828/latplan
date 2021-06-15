#!/usr/bin/env python3

import latplan.main
from latplan.util import *

gs_annealing_epoch = 1000
main_epoch         = 1000
kl_annealing_epoch = 0

# parameters : a dictionary of { name : [ *values ] or value }.
# If the value is a list, it is interpreted as a hyperparameter choice.
# If it is a non-list, single value, it is interpreted as a fixed hyperparameter.

parameters = {
    'test_noise'      : False,   # if true, noise is added during testing
    'test_hard'       : True,    # if true, latent output is discrete
    'train_noise'     : True,    # if true, noise is added during training
    'train_hard'      : False,   # if true, latent output is discrete
    'dropout_z'       : False,
    'noise'           :0.2,
    'dropout'         :0.2,
    'optimizer'       :"radam",
    'min_temperature' :0.5,
    'epoch'           :gs_annealing_epoch+main_epoch+kl_annealing_epoch,
    'gs_annealing_start':0,
    'gs_annealing_end'  :gs_annealing_epoch,
    'kl_cycle_start'    :gs_annealing_epoch+main_epoch,
    'clipnorm'        :0.1,
    'batch_size'      :[400],
    'lr'              :[0.001],
    'N'               :[100], # latent space size
    'zerosuppress'    :0.1,
    'densify'         :False,

    'max_temperature' : [5.0],

    # hyperparameters for encoder/decoder.
    # Each specific class depends only on a subset of hyperparameters.
    # For example, CubeSpaceAE_AMA3Conv uses the hyperparameters for convolutional encoder/decoder only,
    # ignoring the hyperparameters for fully-connected encoder/decoder.
    # Unused hyperparameters are still recorded, but it does not affect the network.
    
    # convolutional
    'conv_channel'              :[32],
    'conv_channel_increment'    :[1],
    'conv_kernel'               :[5],
    'conv_pooling'              :[1], # no pooling
    'conv_per_pooling'          :[1],
    'conv_depth'                :[3], # has_conv_layer = True; so just one convolution

    # fully connected
    'fc_width'       :[100],
    'fc_depth'       :[2],


    # aae
    'A'              :[6000],
    'aae_activation' :['relu'],
    'aae_width'      :[1000],
    'aae_depth'      :[2],

    'eff_regularizer':[None],
    'beta_d'         :[ 1 ],
    'beta_z'         :[ 1 ],

    "output"          :"GaussianOutput(sigma=0.1)",
}

if __name__ == '__main__':
    latplan.main.main(parameters)


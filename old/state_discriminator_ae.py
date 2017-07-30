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
# State discriminator using Autoencoder.
# 
# Perhaps checking the error function |x-encode(decode(x))| may make a good discriminator.

if __name__ == '__main__':
    import numpy.random as random
    from trace import trace

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    
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

    def ds(xs):
        return \
            abs(ae.encode_binary(
                  ae.decode_binary(xs, batch_size=1000), batch_size=1000)
                - xs).max(axis=1)

    ds_valid   = ds(states_valid)
    ds_invalid = ds(states_invalid)
    type_1 = len(np.where(ds_valid   > 0.1)[0])
    type_2 = len(np.where(ds_invalid < 0.1)[0])
    print("type 1 error: {}/{}, {}% (valid states identified as invalid)".format(type_1, len(states_valid), type_1 / len(states_valid) * 100))
    print("type 2 error: {}/{}, {}% (invalid states identified as valid)".format(type_2, len(states_valid), type_2 / len(states_valid) * 100))
    
"""

* Summary:
Input: valid states and invalid states (random bitstrings).

We expect the well-trained AE should be able to decode a bitstring z into an image x and encode x back to z
if z is from the valid examples, and not otherwise.

valid (362880, 28)
invalid (362200, 28)
type 1 error: 17856 (valid states identified as invalid)
type 2 error: 2550  (invalid states identified as valid)

So we expect we can enumerate all bitstrings and choose only the reconstructable states.

"""

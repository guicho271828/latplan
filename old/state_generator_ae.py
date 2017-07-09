#!/usr/bin/env python3
import warnings
import config
import numpy as np
from model import default_networks

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################


from plot import plot_ae

def main():
    import numpy.random as random
    from trace import trace

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    ae = default_networks['fc'](directory).load()
    name = "generated_states.csv"
    
    N = ae.parameters['N']
    lowbit  = 18
    highbit = N - lowbit
    print("batch size: {}".format(2**lowbit))
    
    xs   = (((np.arange(2**lowbit )[:,None] & (1 << np.arange(N)))) > 0).astype(np.int8)
    
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            for i in range(2**highbit):
                print("Iteration {}/{} base: {}".format(i,2**highbit,i*(2**lowbit)), end=' ')
                xs_h = (((np.array([i])[:,None] & (1 << np.arange(highbit)))) > 0).astype(np.int8)
                xs[:,lowbit:] = xs_h
                # print(xs_h)
                # print(xs[:10])
                # ds = ae.autodecode_error_binary(xs,batch_size=100000)
                ds = abs(ae.encode_binary(ae.decode_binary(xs,batch_size=5000),batch_size=5000) - xs).max(axis=1)
                ind = np.where(ds < 0.1)
                valid_xs = xs[ind[0],:]
                print(len(valid_xs))
                np.savetxt(f,valid_xs,"%d",delimiter=" ")
    except KeyboardInterrupt:
        print("dump stopped")

if __name__ == '__main__':
    main()
    
    
"""

* Summary:

Dump all states classified as valid by a discriminator.

Input: all bitstrings (2^N)

Discriminator tells if the state is valid or not.

In this script we use a trained autoencoder and use the reconstruction error as the discrimination signal.

"""

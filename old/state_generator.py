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


from plot import plot_ae

def main():
    import numpy.random as random
    from trace import trace

    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    directory_sd = "{}/_sd/".format(directory)
    sd = Discriminator(directory_sd).load()
    name = "generated_states.csv"
    
    N = sd.net.input_shape[1]
    lowbit  = 20
    highbit = N - lowbit
    print("batch size: {}".format(2**lowbit))
    
    xs   = (((np.arange(2**lowbit )[:,None] & (1 << np.arange(N)))) > 0).astype(int)
    
    try:
        print(sd.local(name))
        with open(sd.local(name), 'wb') as f:
            for i in range(2**highbit):
                print("Iteration {}/{} base: {}".format(i,2**highbit,i*(2**lowbit)), end=' ')
                xs_h = (((np.array([i])[:,None] & (1 << np.arange(highbit)))) > 0).astype(int)
                xs[:,lowbit:] = xs_h
                # print(xs_h)
                # print(xs[:10])
                ys = sd.discriminate(xs,batch_size=100000)
                ind = np.where(ys > 0.8)
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

Discriminator tells if the state is valid or not. (see state_discriminator, error rate 99%)

Still takes 1.4 day, "feasible" but not quite useful for larger problems!

??? Store the states which are shown as valid into a bdd/zdd using graphillion
??? For the effect, use binate algebra

"""

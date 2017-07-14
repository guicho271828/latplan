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
    directory_ad = "{}_ad/".format(directory)
    discriminator = Discriminator(directory_ad).load()
    name = "generated_actions.csv"
    
    N = discriminator.net.input_shape[1]
    lowbit  = 20
    highbit = N - lowbit
    print("batch size: {}".format(2**lowbit))
    
    xs   = (((np.arange(2**lowbit )[:,None] & (1 << np.arange(N)))) > 0).astype(int)
    # xs_h = (((np.arange(2**highbit)[:,None] & (1 << np.arange(highbit)))) > 0).astype(int)
    
    try:
        print(discriminator.local(name))
        with open(discriminator.local(name), 'wb') as f:
            for i in range(2**highbit):
                print("Iteration {}/{} base: {}".format(i,2**highbit,i*(2**lowbit)), end=' ')
                # h = np.binary_repr(i*(2**lowbit), width=N)
                # print(h)
                # xs_h = np.unpackbits(np.array([i*(2**lowbit)],dtype=int))
                xs_h = (((np.array([i])[:,None] & (1 << np.arange(highbit)))) > 0).astype(int)
                xs[:,lowbit:] = xs_h
                # print(xs_h)
                # print(xs[:10])
                ys = discriminator.discriminate(xs,batch_size=100000)
                ind = np.where(ys > 0.5)
                valid_xs = xs[ind]
                print(len(valid_xs))
                np.savetxt(f,valid_xs,"%d")
    except KeyboardInterrupt:
        print("dump stopped")

if __name__ == '__main__':
    main()
    
    
"""
XXX not viable.


* Summary:

Dump all actions classified as valid by a discriminator.

Input: all bitstrings (2^(2*N))

Discriminator tells if the action is valid or not. (see action_discriminator, error rate 99%)

??? Store the actions which are shown as valid into a bdd/zdd using graphillion
??? For the effect, use binate algebra

As expected, this is infeasible for larger latent space; N=36 means 2^72 actions.
Some pruning is required
"""

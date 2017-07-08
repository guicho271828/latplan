#!/usr/bin/env python3
import warnings
import config
import numpy as np
from model import default_networks

import keras.backend as K
import tensorflow as tf

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

from plot import plot_ae, plot_grid, plot_grid2

def main():
    import sys
    if len(sys.argv) == 1:
        sys.exit("{} [directory]".format(sys.argv[0]))

    directory = sys.argv[1]
    ae = default_networks['fc'](directory).load()

    print("loading {}".format("{}/generated_states.csv".format(directory)), end='...', flush=True)
    states = np.loadtxt(ae.local("generated_states.csv"),dtype=np.uint8)
    print("done.")
    zs      = states.view()
    total   = states.shape[0]
    N       = states.shape[1]
    batch   = 500000
    name = "generated_states_unique.csv"
    try:
        print(ae.local(name))
        with open(ae.local(name), 'wb') as f:
            print("original states:",total)
            for i in range(total//batch+1):
                _zs = zs[i*batch:(i+1)*batch]
                _xs = ae.decode_downsample_binary(_zs,batch_size=5000).round().astype(np.uint8)
                _xs_unique, _indices = np.unique(_xs,axis=0,return_index=True)
                _reduced = len(_xs_unique)
                print("reduced  states:",_reduced,"/",len(_zs))
                plot_grid(_xs_unique[:100],w=10,path="generated_states{}.png".format(i))
                _zs_unique = _zs[_indices]
                np.savetxt(f,_zs_unique,"%d",delimiter=" ")
                
    except KeyboardInterrupt:
        print("dump stopped")

if __name__ == '__main__':
    main()
    
    
"""

* Summary:

Remove the duplicate states that results in the same image.

Input: all "valid" bitstrings according to state_generator_ae.

Decode the bitstring, take the a-hash by downsampling and collect the unique states.

"""

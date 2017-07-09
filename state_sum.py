#!/usr/bin/env python3
import warnings
import config
import numpy as np
import model

import keras.backend as K

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################
# to see if there are any biases in the latent space

np.set_printoptions(threshold=np.inf)

def main(path):
    data = np.loadtxt(path,dtype=np.int8)
    
    print(data.sum(axis=0) / len(data))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])

    # 13 bit flips at most

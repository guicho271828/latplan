#!/usr/bin/env python3
import warnings
import config
import numpy as np
import model

import keras.backend as K

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

################################################################

np.set_printoptions(threshold=np.inf)

def main(path):
    data = np.loadtxt(path,dtype=np.int8)

    N = data.shape[1]//2
    pre, suc = data[:,:N], data[:,N:]

    abs_diff = np.sum(np.abs(pre-suc),axis=1)

    hist = np.histogram(abs_diff,N,(0,N))

    print(hist[0])

if __name__ == '__main__':
    import sys
    main(sys.argv[1])

    # 13 bit flips at most

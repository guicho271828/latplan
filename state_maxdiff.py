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

    N = data.shape[1]
    batch = data.shape[0]
    hist = np.zeros(N,dtype=int)

    for i in range(batch):
        abs_diff = np.sum(np.abs(data-data[i],dtype=int),axis=1)
        hist += np.histogram(abs_diff,N,(0,N))[0]
        print(i, hist)


if __name__ == '__main__':
    import sys
    main(sys.argv[1])

    # 13 bit flips at most

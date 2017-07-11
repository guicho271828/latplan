#!/usr/bin/env python3

import numpy as np

def mnist (labels = range(10)):
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype('float32') / 255.).round()
    x_test = (x_test.astype('float32') / 255.).round()
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    def conc (x,y):
        return np.concatenate((y.reshape([len(y),1]),x),axis=1)
    def select (x,y):
        selected = np.array([elem for elem in conc(x, y) if elem[0] in labels])
        return np.delete(selected,0,1), np.delete(selected,np.s_[1::],1).flatten()
    x_train, y_train = select(x_train, y_train)
    x_test, y_test = select(x_test, y_test)
    return x_train, y_train, x_test, y_test


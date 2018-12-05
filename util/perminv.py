
from keras.layers import *

perminv_counts=[-1,-1,-1]
def PermInv1(output_features, activation='relu'):
    import keras.activations as activations
    activation = activations.get(activation)
    lmbda = Convolution1D(output_features, 1, bias=False)
    gamma = Convolution1D(output_features, 1, bias=False)

    def call(x):
        return activation(lmbda(x) + gamma(K.sum(x, axis=1, keepdims=True)))
    perminv_counts[0]+=1
    return Lambda(call,name='PermInv1_{}'.format(perminv_counts[0]))

def PermInv2(output_features, activation='relu'):
    import keras.activations as activations
    activation = activations.get(activation)
    lmbda = Convolution1D(output_features, 1, bias=False)
    gamma = Convolution1D(output_features, 1, bias=False)
    
    def call(x):
        return activation(lmbda(x) - gamma(K.max(x, axis=1, keepdims=True)))
    perminv_counts[1]+=1
    return Lambda(call,name='PermInv2_{}'.format(perminv_counts[1]))

def PermInv3(output_features, activation='relu'):
    gamma = Convolution1D(output_features, 1, bias=True, activation=activation)
    def call(x):
        return gamma(x - K.max(x, axis=1, keepdims=True))
    perminv_counts[2]+=1
    return Lambda(call,name='PermInv3_{}'.format(perminv_counts[2]))

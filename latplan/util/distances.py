
import keras.backend.tensorflow_backend as K

from keras.objectives import categorical_crossentropy as cce
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse, mae

def bcce(true, pred):
    if K.ndim(true) == 3:
        return cce(K.clip(true, 1e-8, 1-1e-8), K.clip(pred, 1e-8, 1-1e-8))
    else:
        return bce(K.clip(true, 1e-8, 1-1e-8), K.clip(pred, 1e-8, 1-1e-8))

def BCE(x, y):
    return bce(K.batch_flatten(x),
               K.batch_flatten(y))
def MSE(x, y):
    return mse(K.batch_flatten(x),
               K.batch_flatten(y))
def MAE(x, y):
    return mae(K.batch_flatten(x),
               K.batch_flatten(y))

def BCE2(x, y):
    "Do not average over bits"
    return K.sum(K.binary_crossentropy(K.batch_flatten(x), K.batch_flatten(y)), axis=-1)

def SE(x, y):
    "Square Error"
    return K.sum(K.square(K.batch_flatten(x) - K.batch_flatten(y)), axis=-1)


# Symmetric Cross Entropy for Robust Learning with Noisy Labels
# ICCV 2019
def SymmetricBCE(x, y):
    return BCE(x,y) + BCE(y,x)

def SymmetricBCE_hard(x, y):
    x2 = x
    x2 = K.round(x2)
    x2 = K.stop_gradient(x2)
    y2 = y
    y2 = K.round(y2)
    y2 = K.stop_gradient(y2)
    return BCE(x2,y) + BCE(y2,x)


# Hausdorff distance
# (Piramuthu 1999) The Hausdor Distance Measure for Feature Selection in Learning Applications
# Hausdorff distance is defined by sup_x inf_y d(x,y).

# algorithm: repeat both x and y N times for N objects.
# pros: matrix operation, fast.
# cons: N^2 memory

def _repeat_nxn_matrix(x,y,N):
    y2 = K.expand_dims(y,1)          # [batch, 1, N, features]
    y2 = K.repeat_elements(y2, N, 1) # [batch, N, N, features]
    # results in [[y1,y2,y3...],[y1,y2,y3...],...]
    
    x2 = K.repeat_elements(x, N, 1)  # [batch, N*N, features]
    x2 = K.reshape(x2, K.shape(y2))  # [batch, N, N, features]
    # results in [[x1,x1,x1...],[x2,x2,x2...],...]
    
    return x2, y2

def Hausdorff(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)

    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sup_x = K.max(K.min(d, axis=2), axis=1) # [batch]
    sup_y = K.max(K.min(d, axis=1), axis=1) # [batch] --- mind the axis
    return K.mean(K.maximum(sup_x, sup_y))

def DirectedHausdorff1(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)

    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sup_x = K.max(K.min(d, axis=2), axis=1) # [batch]
    return K.mean(sup_x)

def DirectedHausdorff2(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)

    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sup_y = K.max(K.min(d, axis=1), axis=1) # [batch] --- mind the axis
    return K.mean(sup_y)

# average distance: (Fujita 2013) Metrics based on average distance between sets

def SumMin(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)
    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sum_x = K.sum(K.min(d, axis=2), axis=1) # [batch]
    sum_y = K.sum(K.min(d, axis=1), axis=1) # [batch] --- mind the axis
    return K.mean(K.maximum(sum_x, sum_y))

def DirectedSumMin1(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)
    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sum_x = K.sum(K.min(d, axis=2), axis=1) # [batch]
    return K.mean(sum_x)

def DirectedSumMin2(distance, x, y, N):
    x2, y2 = _repeat_nxn_matrix(x,y,N)
    d  = K.sum(distance(x2,y2), axis=-1) # [batch, N, N]

    sum_y = K.sum(K.min(d, axis=1), axis=1) # [batch] --- mind the axis
    return K.mean(sum_y)

def set_BCE(x, y, N, combine=DirectedSumMin1):
    return combine(K.binary_crossentropy,x,y,N)

def set_MSE(x, y, N, combine=DirectedSumMin1):
    return combine(lambda x,y: K.square(x-y),x,y,N)


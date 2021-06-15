
import numpy as np

# numpy-based losses
def bce(true,pred,axis=None,epsilon=1e-7):
    x = true
    y = pred
    result = - ( (  x   * np.log(np.clip(y,  epsilon,1))) + \
                 ((1-x) * np.log(np.clip(1-y,epsilon,1)))   ).mean(axis=axis)
    if result.size == 1:
        return float(result)
    else:
        return result

def mae(x,y,axis=None):
    result = np.mean(np.absolute(x - y),axis=axis)
    if result.size == 1:
        return float(result)
    else:
        return result

def mse(x,y,axis=None):
    result = np.mean(np.square(x - y),axis=axis)
    if result.size == 1:
        return float(result)
    else:
        return result

def inf(x,y,axis=None):
    result = np.max(np.abs(x - y),axis=axis)
    if result.size == 1:
        return float(result)
    else:
        return result

# def bce(x,y):
#     from keras.layers import Input
#     from keras.models import Model
#     i = Input(shape=x.shape[1:])
#     m = Model(i,i)
#     m.compile(optimizer="adam", loss='binary_crossentropy')
#     return m.evaluate(x,y,batch_size=1000,verbose=0)
# 
# def mae(x,y):
#     from keras.layers import Input
#     from keras.models import Model
#     i = Input(shape=x.shape[1:])
#     m = Model(i,i)
#     m.compile(optimizer="adam", loss='mean_absolute_error')
#     return m.evaluate(x,y,batch_size=1000,verbose=0)

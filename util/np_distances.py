
import numpy as np

# numpy-based losses
def bce(true,pred,axis=None,epsilon=1e-7):
    x = true
    y = pred
    return - (x * np.log(np.clip(y,epsilon,1)) + \
              (1-x) * np.log(np.clip(1-y,epsilon,1))).mean(axis=axis)

def mae(x,y,axis=None):
    return np.mean(np.absolute(x - y),axis=axis)

def mse(x,y,axis=None):
    return np.sqrt(np.sum(np.square(x - y),axis=axis))


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


from keras.layers import Lambda
def wrap(x,y,**kwargs):
    "wrap arbitrary operation"
    return Lambda(lambda x:y,**kwargs)(x)


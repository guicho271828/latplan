
import keras
from keras.models import Model
from keras.layers import *
from latplan.util.layers import Densify

x = keras.layers.Input(shape=(100,))

d = Densify([
    Dense(100),
    Dense(100),
    Dense(100)
])

y = d(x)

m = Model(x,y)
m.summary()

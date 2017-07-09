import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Dropout, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise, LocallyConnected2D, merge
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
from keras import backend as K
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse, mae
from model import Network

class Generator(Network):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)

    def _build(self,input_shape):
        discriminator, loss = self.parameters['discriminator']
        if discriminator.trainable:
            print("discriminator is set to untrainable")
            discriminator.trainable = False

        x = Input(input_shape)  # assumes zero vector
        generated = Sequential([
            Lambda(lambda x: return x + K.random_uniform(shape=input_shape))
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(np.prod(input_shape),activation="sigmoid"),
            Reshape(input_shape)
        ])(x)

        discriminator_output = discriminator(generated)
        
        self._discriminator = discriminator
        self._generator     = Model(x, generated)
        self.net            = Model(x, discriminator_output)
        self.loss           = loss
        
    def _save(self):
        super()._save()
        self.net.save_weights(self.local("net.h5"))
        
    def _load(self):
        super()._load()
        self.net.load_weights(self.local("net.h5"))
        
    # def report(self,train_data,
    #            epoch=200,
    #            batch_size=1000,optimizer=Adam(0.001),
    #            test_data=None,
    #            train_data_to=None,
    #            test_data_to=None,):
    #     opts = {'verbose':0,'batch_size':batch_size}
    #     def test_both(msg, fn): 
    #         print(msg.format(fn(train_data,train_data_to)))
    #         if test_data is not None:
    #             print((msg+" (validation)").format(fn(test_data,test_data_to)))
    #     self.net.compile(optimizer=optimizer, loss=mae)
    #     test_both("MAE: {}",
    #               lambda data, data_to: self.net.evaluate(data,data_to,**opts))
    #     return self
    def generate(self,data,**kwargs):
        self.load()
        return self._generator.predict(data,**kwargs)
    
    def summary(self,verbose=False):
        self.net.summary()
        return self


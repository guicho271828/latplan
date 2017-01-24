#!/usr/bin/env python3

import numpy as np
from functools import reduce
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse
from keras.callbacks import LambdaCallback
from keras.regularizers import activity_l2, activity_l1


def Sequential (array):
    def apply1(arg,f):
        return f(arg)
    return lambda x: reduce(apply1, array, x)

def Residual (layer):
    def res(x):
        return x+layer(x)
    return Lambda(res)

def ResUnit (*layers):
    return Residual(
        Sequential(layers))

class Network:
    def __init__(self,path,parameters={}):
        import subprocess
        subprocess.call(["mkdir",path])
        self.path = path
        self.built = False
        self.loaded = False
        self.verbose = True
        self.parameters = parameters
        self.custom_log_functions = {}
        self.callbacks = [LambdaCallback(on_epoch_end=self.bar_update)]
    def build(self,input_shape):
        if self.built:
            if self.verbose:
                print("Avoided building {} twice.".format(self))
            return
        self._build(input_shape)
        self.built = True
        return self
    def _build(self):
        pass
    def local(self,path):
        import os.path as p
        return p.join(self.path,path)
    def save(self):
        self._save()
        return self
    def _save(self):
        import json
        with open(self.local('aux.json'), 'w') as f:
            json.dump({"parameters":self.parameters,
                       "input_shape":self.net.input_shape[1:]}, f)
    def load(self):
        if not self.loaded:
            self._load()
            self.loaded = True
        return self
    def _load(self):
        import json
        with open(self.local('aux.json'), 'r') as f:
            data = json.load(f)
            self.parameters = data["parameters"]
            self.build(tuple(data["input_shape"]))
    def bar_update(self, epoch, logs):
        custom_log_values = {}
        for k in self.custom_log_functions:
            custom_log_values[k] = self.custom_log_functions[k]()
        self.bar.update(epoch, **custom_log_values, **logs)
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer=Adam(0.001),test_data=None,save=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        validation = (test_data,test_data_to) if test_data is not None else None
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.build(train_data.shape[1:])
        self.summary()
        print({"parameters":self.parameters,
               "train_shape":train_data.shape,
               "test_shape":test_data.shape})
        try:
            import progressbar
            self.bar = progressbar.ProgressBar(
                max_value=epoch,
                widgets=[
                    progressbar.Timer(format='%(elapsed)s'),
                    progressbar.Bar(),
                    progressbar.AbsoluteETA(format='%(eta)s'), ' ',
                    *(np.array(
                        [ [progressbar.DynamicMessage(k), ' ',]
                          for k in self.custom_log_functions ]).flatten()),
                    progressbar.DynamicMessage('val_loss'), ' ',
                    progressbar.DynamicMessage('loss')
                ]
            )
            self.net.compile(optimizer=optimizer, loss=self.loss)
            self.net.fit(
                train_data, train_data_to,
                nb_epoch=epoch, batch_size=batch_size,
                shuffle=True, validation_data=validation, verbose=False,
                callbacks=self.callbacks)
        except KeyboardInterrupt:
            print("learning stopped")
        self.loaded = True
        self.report(train_data,
                    epoch,batch_size,optimizer,
                    test_data,train_data_to,test_data_to)
        if save:
            self.save()
        return self
    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None):
        pass

class GumbelAE(Network):
    def __init__(self,path,M=2,N=25,parameters={}):
        super().__init__(path,{'M':M,'N':N,**parameters})
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
        self.callbacks.append(LambdaCallback(on_epoch_end=self.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(self.__tau)
    def build_encoder(self,input_shape):
        data_dim = np.prod(input_shape)
        M, N = self.parameters['M'], self.parameters['N']
        return [Reshape((data_dim,)),
                GaussianNoise(0.1),
                Dense(self.parameters['layer'], activation='relu'),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['layer'], activation='relu'),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(M*N),
                Reshape((N,M))]
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        M, N = self.parameters['M'], self.parameters['N']
        return [
            SpatialDropout1D(self.parameters['dropout']),
            # normal dropout after reshape was also effective
            Reshape((N*M,)),
            Dense(self.parameters['layer'], activation='relu'),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu'),
            Dropout(self.parameters['dropout']),
            Dense(data_dim, activation='sigmoid'),
            Reshape(input_shape),]
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N = self.parameters['M'], self.parameters['N']
        tau = K.variable(self.max_temperature, name="temperature")
        def sampling(logits):
            U = K.random_uniform(K.shape(logits), 0, 1)
            z = logits - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            return softmax( z / tau )
        x = Input(shape=input_shape)
        logits = Sequential(self.build_encoder(input_shape))(x)
        z = Lambda(sampling)(logits)
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z)
        z2 = Input(shape=(N,M))
        y2 = Sequential(_decoder)(z2)
        z3 = Lambda(lambda z:K.round(z))(z)
        y3 = Sequential(_decoder)(z3)

        def gumbel_loss(x, y):
            q = softmax(logits)
            log_q = K.log(q + 1e-20)
            kl_loss = K.sum(q * (log_q - K.log(1.0/M)), axis=(1, 2))
            reconstruction_loss = data_dim * bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                                 K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss - kl_loss
        
        self.__tau = tau
        self.loss = gumbel_loss
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.net = Model(x, y)
        self.autoencoder = self.net
        self.autoencoder_binary = Model(x, y3)
    def _save(self):
        super()._save()
        self.encoder.save_weights(self.local("encoder.h5"))
        self.decoder.save_weights(self.local("decoder.h5"))
    def _load(self):
        super()._load()
        self.encoder.load_weights(self.local("encoder.h5"))
        self.decoder.load_weights(self.local("decoder.h5"))
    def cool(self, epoch, logs):
        K.set_value(
            self.__tau,
            np.max([K.get_value(self.__tau) * np.exp(- self.anneal_rate * epoch),
                    self.min_temperature]))
    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {'verbose':0,'batch_size':batch_size}
        def test_both(msg, fn):
            print(msg.format(fn(train_data)))
            if test_data is not None:
                print((msg+" (validation)").format(fn(test_data)))
        self.autoencoder.compile(optimizer=optimizer, loss=mse)
        test_both("Reconstruction MSE: {}",
                  lambda data: self.autoencoder.evaluate(data,data,**opts))
        self.autoencoder_binary.compile(optimizer=optimizer, loss=mse)
        test_both("Binary Reconstruction MSE: {}",
                  lambda data: self.autoencoder_binary.evaluate(data,data,**opts))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        test_both("Reconstruction BCE: {}",
                  lambda data: self.autoencoder.evaluate(data,data,**opts))
        self.autoencoder_binary.compile(optimizer=optimizer, loss=bce)
        test_both("Binary Reconstruction BCE: {}",
                  lambda data: self.autoencoder_binary.evaluate(data,data,**opts))
        test_both("Latent activation: {}",
                  lambda data: self.encode_binary(train_data,batch_size=batch_size,).mean())
        return self
    def encode(self,data,**kwargs):
        self.load()
        return self.encoder.predict(data,**kwargs)
    def decode(self,data,**kwargs):
        self.load()
        return self.decoder.predict(data,**kwargs)
    def autoencode(self,data,**kwargs):
        self.load()
        return self.autoencoder.predict(data,**kwargs)
    def encode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.encode(data,**kwargs)[:,:,0].reshape(-1, N)
    def decode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.decode(np.stack((data,1-data),axis=-1),**kwargs)
    def summary(self,verbose=False):
        if verbose:
            self.encoder.summary()
            self.decoder.summary()
        self.autoencoder.summary()
        return self

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import shlex, subprocess
    from mnist import mnist
    x_train, _, x_test, _ = mnist()
    ae = GumbelAE("samples/mnist_model/")
    ae.train(x_train,test_data=x_test)
    ae.summary()
    del ae
    ae = GumbelAE("samples/mnist_model/")
    howmany=10
    y_test = ae.autoencode(x_test[:howmany])
    z_test = ae.encode_binary(x_test[:howmany])

    plt.figure(figsize=(30, 5))
    for i in range(howmany):
        plt.subplot(3,howmany,i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(3,howmany,i+1+1*howmany)
        plt.imshow(z_test[i].reshape(4,4), cmap='gray',
                   interpolation='nearest')
        plt.axis('off')
        plt.subplot(3,howmany,i+1+2*howmany)
        plt.imshow(y_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.savefig(ae.local('viz.png'))


class ConvolutionalGumbelAE(GumbelAE):
    def build_encoder(self,input_shape):
        data_dim = np.prod(input_shape)
        M, N = self.parameters['M'], self.parameters['N']
        # Trainable parameters: 1,436,320
        return [Reshape(input_shape+(1,)),
                GaussianNoise(0.1),
                Convolution2D(128,6,5,activation='relu',border_mode='same'),
                SpatialDropout2D(0.4),
                BN(),
                Convolution2D(128,6,5,subsample=(6,5),activation='relu',border_mode='same'),
                SpatialDropout2D(0.4),
                BN(),
                Flatten(),
                Dense(M*N),
                Reshape((N,M))]

class Discriminator(Network):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
        self.callbacks.append(LambdaCallback(on_epoch_end=self.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(self.__tau)
    def build_encoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [Reshape((data_dim,)),
                Dense(self.parameters['layer'], activation='relu'),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['layer'], activation='relu'),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(2)]
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        tau = K.variable(self.max_temperature, name="temperature")
        def sampling(logits):
            U = K.random_uniform(K.shape(logits), 0, 1)
            z = logits - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            return softmax( z / tau )
        x = Input(shape=input_shape)
        logits = Sequential(self.build_encoder(input_shape))(x)
        z = Lambda(sampling)(logits)
        z3 = Lambda(lambda z:K.round(z))(z)

        def gumbel_loss(x, y):
            q = softmax(logits)
            log_q = K.log(q + 1e-20)
            kl_loss = K.sum(q * (log_q - K.log(1.0/2)), axis=1)
            reconstruction_loss = data_dim * bce(x,y)
            return reconstruction_loss - kl_loss
        
        self.__tau = tau
        self.loss = gumbel_loss
        self.net = Model(x, z)
        self.net_binary = Model(x, z3)
    def _save(self):
        super()._save()
        self.net.save_weights(self.local("net.h5"))
    def _load(self):
        super()._load()
        self.net.load_weights(self.local("net.h5"))
    def cool(self, epoch, logs):
        K.set_value(
            self.__tau,
            np.max([K.get_value(self.__tau) * np.exp(- self.anneal_rate * epoch),
                    self.min_temperature]))
    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {'verbose':0,'batch_size':batch_size}
        def test_both(msg, fn): 
            print(msg.format(fn(train_data,train_data_to)))
            if test_data is not None:
                print((msg+" (validation)").format(fn(test_data,test_data_to)))
        self.net.compile(optimizer=optimizer, loss=bce)
        test_both("BCE: {}",
                  lambda data, data_to: self.net.evaluate(data,data_to,**opts))
        self.net_binary.compile(optimizer=optimizer, loss=bce)
        test_both("Binary BCE: {}",
                  lambda data, data_to: self.net_binary.evaluate(data,data_to,**opts))
        return self
    def discriminate(self,data,**kwargs):
        self.load()
        return self.net.predict(data,**kwargs)
    def discriminate_binary(self,data,**kwargs):
        return self.discriminate(data,**kwargs)[:,:,0].reshape(-1, N)
    def summary(self,verbose=False):
        self.net.summary()
        return self
    

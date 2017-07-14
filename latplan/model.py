#!/usr/bin/env python3

"""
Networks named like XXX2 uses gumbel softmax for the output layer too,
assuming the input/output image is binarized
"""

import numpy as np
from functools import reduce
from keras.layers import Input, Dense, Dropout, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise, LocallyConnected2D, merge
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse, mae
from keras.callbacks import LambdaCallback, LearningRateScheduler
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf

# utilities ###############################################################
def wrap(x,y,**kwargs):
    "wrap arbitrary operation"
    return Lambda(lambda x:y,**kwargs)(x)

def flatten(x):
    if K.ndim(x) >= 3:
        return Flatten()(x)
    else:
        return x
    
def Print():
    def printer(x):
        print(x)
        return x
    return Lambda(printer)

def Sequential (array):
    def apply1(arg,f):
        # print("applying {}({})".format(f,arg))
        return f(arg)
    return lambda x: reduce(apply1, array, x)

def Residual (layer):
    def res(x):
        return x+layer(x)
    return Lambda(res)

def ResUnit (*layers):
    return Residual(
        Sequential(layers))

from keras.constraints import Constraint, maxnorm,nonneg,unitnorm
class UnitNormL1(Constraint):
    def __init__(self, axis=0):
        self.axis = axis

    def __call__(self, p):
        return p / (K.epsilon() + K.sum(p,
                                        axis=self.axis,
                                        keepdims=True))

    def get_config(self):
        return {'name': self.__class__.__name__,
                'axis': self.axis}

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
        print("Saving to {}".format(self.local('')))
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
        self.net.compile(Adam(0.0001),bce)
        
    def bar_update(self, epoch, logs):
        ologs = {}
        for k in self.custom_log_functions:
            ologs[k] = self.custom_log_functions[k]()
        for k in logs:
            if len(k) > 5:
                ologs[k[-5:]] = logs[k]
            else:
                ologs[k] = logs[k]

        if not hasattr(self,'bar'):
            import progressbar
            widgets = [
                progressbar.Timer(format='%(elapsed)s'),
                ' ', progressbar.Counter(), 
                progressbar.Bar(),
                progressbar.AbsoluteETA(format='%(eta)s'), ' ',
            ]
            keys = []
            for k in ologs:
                keys.append(k)
            keys.sort()
            for k in keys:
                widgets.append(progressbar.DynamicMessage(k))
                widgets.append(' ')
            self.bar = progressbar.ProgressBar(max_value=self.max_epoch, widgets=widgets)
        self.bar.update(epoch+1, **ologs)
        
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer=Adam,lr=0.0001,test_data=None,save=True,report=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
        o = optimizer(lr)
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to

        for k,v in kwargs.items():
            setattr(self, k, v)
        self.build(train_data.shape[1:])
        self.summary()
        print({"parameters":self.parameters,
               "train_shape":train_data.shape,
               "test_shape":test_data.shape})

        if isinstance(self.loss,list):
            train_data_to = [ train_data_to for l in self.loss ]
            test_data_to = [ test_data_to for l in self.loss ]
        validation = (test_data,test_data_to) if test_data is not None else None
        try:
            self.max_epoch = epoch
            self.net.compile(optimizer=o, loss=self.loss)
            self.net.fit(
                train_data, train_data_to,
                epochs=epoch, batch_size=batch_size,
                shuffle=True, validation_data=validation, verbose=False,
                callbacks=self.callbacks)
        except KeyboardInterrupt:
            print("learning stopped\n")
        self.loaded = True
        if report:
            self.report(train_data,
                        epoch,batch_size,o,
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

def anneal_rate(epoch,min=0.1,max=5.0):
    import math
    return math.log(max/min) / epoch

def take_true(y_cat):
    return tf.slice(y_cat,[0,0,0],[-1,-1,1])

class GumbelSoftmax:
    count = 0
    
    def __init__(self,N,M,min,max,anneal_rate):
        self.N = N
        self.M = M
        self.layers = Sequential([
            # Dense(N * M),
            Reshape((N,M))])
        self.min = min
        self.max = max
        self.anneal_rate = anneal_rate
        self.tau = K.variable(max, name="temperature")
        
    def call(self,logits):
        u = K.random_uniform(K.shape(logits), 0, 1)
        gumbel = - K.log(-K.log(u + 1e-20) + 1e-20)
        return K.softmax( ( logits + gumbel ) / self.tau )
    
    def __call__(self,prev):
        if hasattr(self,'logits'):
            raise ValueError('do not reuse the same GumbelSoftmax; reuse GumbelSoftmax.layers')
        GumbelSoftmax.count += 1
        c = GumbelSoftmax.count-1
        prev = flatten(prev)
        logits = self.layers(prev)
        self.logits = logits
        return Lambda(self.call,name="gumbel_{}".format(c))(logits)
    
    def loss(self):
        logits = self.logits
        q = K.softmax(logits)
        log_q = K.log(q + 1e-20)
        return - K.mean(q * (log_q - K.log(1.0/K.int_shape(logits)[-1])),
                        axis=tuple(range(1,len(K.int_shape(logits)))))
    
    def cool(self, epoch, logs):
        K.set_value(
            self.tau,
            np.max([self.min,
                    self.max * np.exp(- self.anneal_rate * epoch)]))

class SimpleGumbelSoftmax(GumbelSoftmax):
    "unlinke GumbelSoftmax, it does not have layers."
    count = 0
    def __init__(self,min,max,anneal_rate):
        self.min = min
        self.anneal_rate = anneal_rate
        self.tau = K.variable(max, name="temperature")
    
    def __call__(self,prev):
        if hasattr(self,'logits'):
            raise ValueError('do not reuse the same GumbelSoftmax; reuse GumbelSoftmax.layers')
        SimpleGumbelSoftmax.count += 1
        c = SimpleGumbelSoftmax.count-1
        self.logits = prev
        return Lambda(self.call,name="simplegumbel_{}".format(c))(prev)

# Network mixins ################################################################
class AE(Network):
    def _save(self):
        super()._save()
        self.encoder.save_weights(self.local("encoder.h5"))
        self.decoder.save_weights(self.local("decoder.h5"))
        
    def _load(self):
        super()._load()
        self.encoder.load_weights(self.local("encoder.h5"))
        self.decoder.load_weights(self.local("decoder.h5"))

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
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        test_both("Reconstruction BCE: {}",
                  lambda data: self.autoencoder.evaluate(data,data,**opts))
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

    def summary(self,verbose=False):
        if verbose:
            self.encoder.summary()
            self.decoder.summary()
        self.autoencoder.summary()
        return self

class GumbelAE(AE):
    def build_encoder(self,input_shape):
        return [GaussianNoise(0.1),
                Dense(self.parameters['layer'], activation='relu', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['layer'], activation='relu', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['layer'], activation='relu', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['N']*self.parameters['M']),]
    
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *([Dropout(self.parameters['dropout'])] if self.parameters['dropout_z'] else []),
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(data_dim, activation='sigmoid'),
            Reshape(input_shape),]

    def __init__(self,path,parameters={}):
        if 'N' not in parameters:
            parameters['N'] = 25
        if 'M' not in parameters:
            parameters['M'] = 2
        super().__init__(path,parameters)
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
   
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N = self.parameters['M'], self.parameters['N']
        x = Input(shape=input_shape)
        x_flat = flatten(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        print(Model(x,pre_encoded))
        gs = GumbelSoftmax(N,M,self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(pre_encoded)
        z_flat = flatten(z)
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z_flat)
        
        z2 = Input(shape=(N,M))
        z2_flat = flatten(z2)
        y2 = Sequential(_decoder)(z2_flat)

        def loss(x, y):
            kl_loss = gs.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gs.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs.tau)
        self.loss = loss
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.net = Model(x, y)
        self.autoencoder = self.net
        y2_downsample = Sequential([
            Reshape((*input_shape,1)),
            MaxPooling2D((2,2))
            ])(y2)
        shape = K.int_shape(y2_downsample)[1:3]
        self.decoder_downsample = Model(z2, Reshape(shape)(y2_downsample))
        
    def encode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.encode(data,**kwargs)[:,:,0].reshape(-1, N)
    
    def decode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.decode(np.stack((data,1-data),axis=-1),**kwargs)

    def decode_downsample(self,data,**kwargs):
        self.load()
        return self.decoder_downsample.predict(data,**kwargs)

    def decode_downsample_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.decode_downsample(np.stack((data,1-data),axis=-1),**kwargs)

    def plot(self,data,path,verbose=False):
        self.load()
        xs = data
        zs = self.encode_binary(xs)
        ys = self.decode_binary(zs)
        bs = np.round(zs)
        bys = self.decode_binary(bs)
        M, N = self.parameters['M'], self.parameters['N']
        import math
        root = math.sqrt(N)
        l1 = math.floor(root)
        if l1*l1 == N:
            _zs = zs.reshape((-1,l1,l1))
            _bs = bs.reshape((-1,l1,l1))
        else:
            l2 = math.ceil(root)
            size = l2*l2
            print(root,size,N) if verbose else None
            _zs = np.concatenate((zs,np.ones((zs.shape[0],size-N))),axis=1).reshape((-1,l2,l2))
            _bs = np.concatenate((bs,np.ones((bs.shape[0],size-N))),axis=1).reshape((-1,l2,l2))
        images = []
        from plot import plot_grid
        for seq in zip(xs, _zs, ys, _bs, bys):
            images.extend(seq)
        plot_grid(images, path=self.local(path), verbose=verbose)
        return xs,zs,ys,bs,bys

class GumbelAE2(GumbelAE):
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *([Dropout(self.parameters['dropout'])] if self.parameters['dropout_z'] else []) ,
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(data_dim*2),]
    
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N = self.parameters['M'], self.parameters['N']
        x = Input(shape=input_shape)
        x_flat = flatten(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        gs = GumbelSoftmax(N,M,self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(pre_encoded)
        z_flat = flatten(z)
        _decoder = self.build_decoder(input_shape)
        y_logit  = Sequential(_decoder)(z_flat)
        gs2 = GumbelSoftmax(data_dim,2,self.min_temperature,self.max_temperature,self.anneal_rate)
        y_cat = gs2(y_logit)
        
        y = Reshape(input_shape)(Lambda(take_true)(y_cat))
            
        z2 = Input(shape=(N,M))
        z2_flat = flatten(z2)
        y2_logit = Sequential(_decoder)(z2_flat)
        gs3 = GumbelSoftmax(data_dim,2,self.min_temperature,self.max_temperature,self.anneal_rate)
        y2_cat = gs3(y2_logit)
        y2 = Reshape(input_shape)(Lambda(take_true)(y2_cat))
        
        def loss(x, y):
            kl_loss = gs.loss() + gs2.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gs.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gs2.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gs3.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs.tau)
        self.loss = loss
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.net = Model(x, y)
        self.autoencoder = self.net
        y2_downsample = Sequential([
            Reshape((*input_shape,1)),
            MaxPooling2D((2,2))
            ])(y2)
        shape = K.int_shape(y2_downsample)[1:3]
        self.decoder_downsample = Model(z2, Reshape(shape)(y2_downsample))

class ConvolutionalGumbelAE(GumbelAE):
    def build_encoder(self,input_shape):
        return [Reshape((*input_shape,1)),
                GaussianNoise(0.1),
                Convolution2D(self.parameters['clayer'],3,3,
                              activation=self.parameters['activation'],border_mode='same', use_bias=False),
                Dropout(self.parameters['dropout']),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters['clayer'],3,3,
                              activation=self.parameters['activation'],border_mode='same', use_bias=False),
                Dropout(self.parameters['dropout']),
                BN(),
                MaxPooling2D((2,2)),
                flatten,
                Dense(self.parameters['layer'], activation=self.parameters['activation'], use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['N']*self.parameters['M']),]
    
class Convolutional2GumbelAE(ConvolutionalGumbelAE):
    def build_decoder(self,input_shape):
        "this function did not converge well. sigh"
        data_dim = np.prod(input_shape)
        last_convolution = 1 + np.array(input_shape) // 4
        first_convolution = last_convolution * 4
        diff = tuple(first_convolution - input_shape)
        crop = [[0,0],[0,0]]
        for i in range(2):
            if diff[i] % 2 == 0:
                for j in range(2):
                    crop[i][j] = diff[i] // 2
            else:
                crop[i][0] = diff[i] // 2
                crop[i][1] = diff[i] // 2 + 1
        crop = ((crop[0][0],crop[0][1]),(crop[1][0],crop[1][1]))
        print(last_convolution,first_convolution,diff,crop)
        
        return [*([Dropout(self.parameters['dropout'])] if self.parameters['dropout_z'] else []),
                Dense(self.parameters['layer'], activation='relu', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(np.prod(last_convolution) * self.parameters['clayer'], activation='relu', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                Reshape((*last_convolution, self.parameters['clayer'])),
                UpSampling2D((2,2)),
                Deconvolution2D(self.parameters['clayer'],3,3,
                                activation='relu',border_mode='same', use_bias=False),
                BN(),
                Dropout(self.parameters['dropout']),
                UpSampling2D((2,2)),
                Deconvolution2D(1,3,3,
                                activation='sigmoid',border_mode='same'),
                Cropping2D(crop),
                Reshape(input_shape),]

# mixin classes ###############################################################
# Now effectively 3 subclasses; GumbelSoftmax in the output, Convolution, Gaussian.
# there are 4 more results of mixins:
class ConvolutionalGumbelAE2(ConvolutionalGumbelAE,GumbelAE2):
    pass

# state/action discriminator ####################################################
class Discriminator(Network):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)

    def _build(self,input_shape):
        x = Input(shape=input_shape)
        N = input_shape[0] // 2

        actions_any = Sequential([
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'],activation=self.parameters['activation']),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(1,activation="sigmoid")
        ])(x)
        self._actions_any = Model(x, actions_any)

        def loss(x,y):
            return bce(x,y)
        self.loss = loss
        self.net = Model(x, actions_any)
        
    def _save(self):
        super()._save()
        self.net.save_weights(self.local("net.h5"))
    def _load(self):
        super()._load()
        self.net.load_weights(self.local("net.h5"))
    def report(self,train_data,
               epoch=200,
               batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        opts = {'verbose':0,'batch_size':batch_size}
        def test_both(msg, fn): 
            print(msg.format(fn(train_data,train_data_to)))
            if test_data is not None:
                print((msg+" (validation)").format(fn(test_data,test_data_to)))
        self.net.compile(optimizer=optimizer, loss=mae)
        test_both("MAE: {}",
                  lambda data, data_to: self.net.evaluate(data,data_to,**opts))
        return self
    def discriminate(self,data,**kwargs):
        self.load()
        return self._actions_any.predict(data,**kwargs)
    def summary(self,verbose=False):
        self.net.summary()
        return self



def main ():
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
    
if __name__ == '__main__':
    main()

default_networks = {'fc':GumbelAE,'conv':ConvolutionalGumbelAE,
                    'fc2':GumbelAE2,'conv2':ConvolutionalGumbelAE2,}

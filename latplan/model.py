#!/usr/bin/env python3

"""
Networks named like XXX2 uses gumbel softmax for the output layer too,
assuming the input/output image is binarized
"""

import numpy as np
from functools import reduce
from keras.layers import Input, Dense, Dropout, Convolution2D, Deconvolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise, LocallyConnected2D, merge
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
import keras.optimizers
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

debug = False
# utilities ###############################################################
def wrap(x,y,**kwargs):
    "wrap arbitrary operation"
    return Lambda(lambda x:y,**kwargs)(x)

def flatten(x):
    if K.ndim(x) >= 3:
        return Flatten()(x)
    else:
        return x

def set_trainable (model, flag):
    if hasattr(model, "layers"):
        for l in model.layers:
            set_trainable(l, flag)
    else:
        model.trainable = flag
    
def Print():
    def printer(x):
        print(x)
        return x
    return Lambda(printer)

def Sequential (array):
    def apply1(arg,f):
        if debug:
            print("applying {}({})".format(f,arg))
        result = f(arg)
        if debug:
            print(K.int_shape(result))
        return result
    return lambda x: reduce(apply1, array, x)

def ConditionalSequential (array, condition, **kwargs):
    def apply1(arg,f):
        # print("applying {}({})".format(f,arg))
        concat = Concatenate(**kwargs)([condition, arg])
        return f(concat)
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
            
    def load(self,allow_failure=False):
        if allow_failure:
            try:
                if not self.loaded:
                    self._load()
                    self.loaded = True
            except Exception as e:
                print("Exception {} during load(), ignored.".format(e))
        else:
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
              epoch=200,batch_size=1000,optimizer='adam',lr=0.0001,test_data=None,save=True,report=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
        o = getattr(keras.optimizers,optimizer)(lr)
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to

        self.build(train_data.shape[1:])
        if debug:
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

    def build_gs(self,
                 **kwargs):
        # hack, but is useful overall

        def fn(N=self.parameters['N'],
               M=self.parameters['M'],
               max_temperature=self.parameters['max_temperature'],
               min_temperature=self.parameters['min_temperature'],
               full_epoch=self.parameters['full_epoch'],):
            return GumbelSoftmax(
                N,M,min_temperature,max_temperature,
                anneal_rate(full_epoch, min_temperature, max_temperature))
        return fn(**kwargs)

class GumbelAE(AE):
    def build_encoder(self,input_shape):
        return [GaussianNoise(0.1),
                BN(),
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

    def _build(self,input_shape):
        self.gs = self.build_gs()

        _encoder = self.build_encoder(input_shape)
        _decoder = self.build_decoder(input_shape)
        
        x = Input(shape=input_shape)
        z = Sequential([flatten, *_encoder, self.gs])(x)
        y = Sequential(_decoder)(flatten(z))
         
        z2 = Input(shape=(self.parameters['N'], self.parameters['M']))
        y2 = Sequential(_decoder)(flatten(z2))

        data_dim = np.prod(input_shape)
        def loss(x, y):
            kl_loss = self.gs.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=self.gs.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(self.gs.tau)
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
        x = data
        z = self.encode_binary(x)
        y = self.decode_binary(z)
        b = np.round(z)
        by = self.decode_binary(b)
        M, N = self.parameters['M'], self.parameters['N']

        from .util.plot import plot_grid, squarify
        _z = squarify(z)
        _b = squarify(b)
        
        images = []
        from .util.plot import plot_grid
        for seq in zip(x, _z, y, y.round(), _b, by, by.round()):
            images.extend(seq)
        plot_grid(images, w=14, path=self.local(path), verbose=verbose)
        return x,z,y,b,by

    def plot_autodecode(self,data,path,verbose=False):
        self.load()
        z = data
        x = self.decode_binary(z)
        z2 = self.encode_binary(x)
        z2r = z2.round()
        x2 = self.decode_binary(z2)
        x2r = self.decode_binary(z2r)
        M, N = self.parameters['M'], self.parameters['N']

        from .util.plot import plot_grid, squarify
        _z = squarify(z)
        _z2 = squarify(z2)
        _z2r = _z2.round()
        
        images = []
        from .util.plot import plot_grid
        for seq in zip(_z, x, _z2, _z2r, x2, x2r):
            images.extend(seq)
        plot_grid(images, w=6, path=self.local(path), verbose=verbose)
        return _z, x, _z2, _z2r

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
        gs = self.build_gs()
        z = gs(pre_encoded)
        z_flat = flatten(z)
        _decoder = self.build_decoder(input_shape)
        y_logit  = Sequential(_decoder)(z_flat)
        gs2 = self.build_gs(N=data_dim)
        y_cat = gs2(y_logit)
        
        y = Reshape(input_shape)(Lambda(take_true)(y_cat))
            
        z2 = Input(shape=(N,M))
        z2_flat = flatten(z2)
        y2_logit = Sequential(_decoder)(z2_flat)
        gs3 = self.build_gs(N=data_dim)
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
                BN(),
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
    def _build(self,input_shape):
        x = Input(shape=input_shape)
        N = input_shape[0] // 2

        actions_any = Sequential([
            *[Sequential([Dense(self.parameters['layer'],activation=self.parameters['activation']),
                          BN(),
                          Dropout(self.parameters['dropout']),])
              for i in range(self.parameters['num_layers']) ],
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


# action autoencoder ################################################################

class ActionAE(AE):
    # A network which autoencodes the difference information.
    # 
    # State transitions are not a 1-to-1 mapping in a sense that
    # there are multiple applicable actions. So you cannot train a newtork that directly learns
    # a transition S -> T .
    # 
    # We also do not have action labels, so we need to cluster the actions in an unsupervised manner.
    # 
    # This network trains a bidirectional mapping of (S,T) -> (S,A) -> (S,T), given that 
    # a state transition is a function conditioned by the before-state s.
    # 
    # It is not useful to learn a normal autoencoder (S,T) -> Z -> (S,T) because we cannot separate the
    # condition and the action label.
    # 
    # We again use gumbel-softmax for representing A.
    # (*undetermined*) To learn the lifted representation,
    # A is a single variable with M categories. We do not specify N.
    def build_encoder(self,input_shape):
        return [
            *[
                Sequential([
                    Dense(self.parameters['layer'], activation=self.parameters['encoder_activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),])
                for i in range(self.parameters['encoder_layers'])
            ],
            Dense(self.parameters['N']*self.parameters['M']),]
    
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters['layer'], activation=self.parameters['decoder_activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),])
                for i in range(self.parameters['decoder_layers'])
            ],
            Sequential([
                Dense(data_dim, activation='sigmoid'),
                Reshape(input_shape),]),]
   
    def _build(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters['M'], self.parameters['N']
        
        x = Input(shape=input_shape)

        _pre = tf.slice(x, [0,0],   [-1,dim])
        _suc = tf.slice(x, [0,dim], [-1,dim])
        
        pre = wrap(x,_pre,name="pre")
        suc = wrap(x,_suc,name="suc")

        print("encoder")
        _encoder = self.build_encoder([dim])
        action_logit = ConditionalSequential(_encoder, pre, axis=1)(suc)
        
        gs = self.build_gs()
        action = gs(action_logit)

        print("decoder")
        _decoder = self.build_decoder([dim])
        suc_reconstruction = ConditionalSequential(_decoder, pre, axis=1)(flatten(action))
        y = Concatenate(axis=1)([pre,suc_reconstruction])
        
        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        suc_reconstruction2 = ConditionalSequential(_decoder, pre2, axis=1)(flatten(action2))
        y2 = Concatenate(axis=1)([pre2,suc_reconstruction2])

        def loss(x, y):
            kl_loss = gs.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],dim*2,)),
                                      K.reshape(y,(K.shape(x)[0],dim*2,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gs.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs.tau)
        self.loss = loss
        self.encoder     = Model(x, [pre,action])
        self.decoder     = Model([pre2,action2], y2)

        self.net = Model(x, y)
        self.autoencoder = self.net
        
    def encode_action(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        return self.encode(data,**kwargs)[1]

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
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        test_both("Reconstruction BCE: {}",
                  lambda data: self.autoencoder.evaluate(data,data,**opts))
        return self
    
    def plot(self,data,path,verbose=False,ae=None):
        self.load()
        dim = data.shape[1] // 2
        x = data
        _, z = self.encode(x) # _ == x
        y = self.decode([x[:,:dim],z])
        b = np.round(z)
        by = self.decode([x[:,:dim],b])

        from .util.plot import plot_grid, squarify
        x_pre, x_suc = squarify(x[:,:dim]), squarify(x[:,dim:])
        y_pre, y_suc = squarify(y[:,:dim]), squarify(y[:,dim:])
        by_pre, by_suc = squarify(by[:,:dim]), squarify(by[:,dim:])
        y_suc_r, by_suc_r = y_suc.round(), by_suc.round()

        if ae:
            x_pre_im, x_suc_im = ae.decode_binary(x[:,:dim]), ae.decode_binary(x[:,dim:])
            y_pre_im, y_suc_im = ae.decode_binary(y[:,:dim]), ae.decode_binary(y[:,dim:])
            by_pre_im, by_suc_im = ae.decode_binary(by[:,:dim]), ae.decode_binary(by[:,dim:])
            y_suc_r_im, by_suc_r_im = ae.decode_binary(y[:,dim:].round()), ae.decode_binary(by[:,dim:].round())
            images = []
            for seq in zip(x_pre_im, x_suc_im, z, y_pre_im, y_suc_im, y_suc_r_im, b, by_pre_im, by_suc_im, by_suc_r_im):
                images.extend(seq)
            plot_grid(images, w=10, path=self.local(path), verbose=verbose)
        else:
            images = []
            for seq in zip(x_pre, x_suc, z, y_pre, y_suc, y_suc_r, b, by_pre, by_suc, by_suc_r):
                images.extend(seq)
            plot_grid(images, w=10, path=self.local(path), verbose=verbose)
        return x,z,y,b,by

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

default_networks = {
    'fc':GumbelAE,
    'fc2':GumbelAE2,
    'conv':ConvolutionalGumbelAE,
    'conv2':ConvolutionalGumbelAE2,
    'cc' : Convolutional2GumbelAE,
    **{
        name: classobj \
        for name, classobj in globals().items() \
        if isinstance(classobj, type)
    }
}

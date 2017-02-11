#!/usr/bin/env python3

import numpy as np
from functools import reduce
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise, LocallyConnected2D, merge
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
        self.net.compile(Adam(0.001),bce)
        
    def bar_update(self, epoch, logs):
        s = ""
        for k in self.custom_log_functions:
            s += "{}: {:5.3g}, ".format(k,self.custom_log_functions[k]())
        for k in logs:
            s += "{}: {:5.3g}, ".format(k,logs[k])
        self.bar.update(epoch+1, stat=s)
        
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer=Adam(0.001),test_data=None,save=True,report=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
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
            import progressbar
            self.bar = progressbar.ProgressBar(
                max_value=epoch,
                widgets=[
                    progressbar.Timer(format='%(elapsed)s'),
                    progressbar.Bar(),
                    progressbar.AbsoluteETA(format='%(eta)s'), ' ',
                    progressbar.DynamicMessage('stat')
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
        if report:
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

class GumbelSoftmax:
    count = 0
    
    def __init__(self,N,M,min,max,anneal_rate):
        self.N = N
        self.M = M
        self.layers = Sequential([
            Dense(N * M),
            Reshape((N,M))])
        self.min = min
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
        if K.ndim(prev) >= 3:
            prev = Flatten()(prev)
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
                    K.get_value(self.tau) * np.exp(- self.anneal_rate * epoch)]))

class GaussianSample:
    count = 0
    
    def __init__(self,G):
        self.G = G
        
    def call(self,args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., std=1.0)
        return mean + K.exp(log_var / 2) * epsilon
    
    def __call__(self,prev):
        GaussianSample.count += 1
        c = GaussianSample.count-1
        if K.ndim(prev) >= 3:
            prev = Flatten()(prev)
        mean    = Dense(self.G,name="gmean_{}".format(c))(prev)
        log_var = Dense(self.G,name="glogvar_{}".format(c))(prev)
        self.mean, self.log_var = mean, log_var
        return Lambda(self.call,name="gaussian_{}".format(c))([mean,log_var])
    
    def loss(self):
        return - 0.5 * K.mean(
            1 + self.log_var - K.square(self.mean) - K.exp(self.log_var),
            axis=-1)
        
class GumbelAE(Network):
    def __init__(self,path,parameters={}):
        if 'N' not in parameters:
            parameters['N'] = 25
        if 'M' not in parameters:
            parameters['M'] = 2
        super().__init__(path,parameters)
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
        
    def build_encoder(self,input_shape):
        return [GaussianNoise(0.1),
                Dense(self.parameters['layer'], activation='relu'),
                BN(),
                Dropout(self.parameters['dropout']),
                Dense(self.parameters['layer'], activation='relu'),
                # !!!! gumbel softmax is a softmax, don't use batchnorm !!!!!
                BN(),
                Dropout(self.parameters['dropout']),]
    
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu'),
            # this BN may be initially bad for val_loss, but is ok for longer epochs
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu'),
            # BN(),
            Dropout(self.parameters['dropout']),
            Dense(data_dim, activation='sigmoid'),
            Reshape(input_shape),]
    
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N = self.parameters['M'], self.parameters['N']
        x = Input(shape=input_shape)
        x_flat = Flatten()(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        print(Model(x,pre_encoded))
        gs = GumbelSoftmax(N,M,self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(pre_encoded)
        z_flat = Flatten()(z)
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z_flat)
        
        z2 = Input(shape=(N,M))
        z2_flat = Flatten()(z2)
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

class GumbelAE2(GumbelAE):
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu'),
            # this BN may be initially bad for val_loss, but is ok for longer epochs
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu'),
            # BN(),
            Dropout(self.parameters['dropout']),]
    
    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N = self.parameters['M'], self.parameters['N']
        x = Input(shape=input_shape)
        x_flat = Flatten()(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        gs = GumbelSoftmax(N,M,self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(pre_encoded)
        z_flat = Flatten()(z)
        _decoder = self.build_decoder(input_shape)
        y_logit  = Sequential(_decoder)(z_flat)
        gs2 = GumbelSoftmax(data_dim,2,self.min_temperature,self.max_temperature,self.anneal_rate)
        y_cat = gs2(y_logit)
        
        def take_true(y_cat):
            import tensorflow as tf
            return tf.slice(y_cat,[0,0,0],[-1,-1,1])
        y = Reshape(input_shape)(Lambda(take_true)(y_cat))
            
        z2 = Input(shape=(N,M))
        z2_flat = Flatten()(z2)
        y2_logit = Sequential(_decoder)(z2_flat)
        gs3 = GumbelSoftmax(data_dim,2,self.min_temperature,self.max_temperature,self.anneal_rate)
        gs3.layers = gs2.layers
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

class ConvolutionalGumbelAE(GumbelAE):
    def build_encoder(self,input_shape):
        return [Reshape((*input_shape,1)),
                GaussianNoise(0.1),
                Convolution2D(9,3,3,subsample=(3,3),
                              activation='relu',border_mode='same'),
                Dropout(self.parameters['dropout']),
                BN(),
                Convolution2D(81,3,3,subsample=(3,3),
                              activation='relu',border_mode='same'),
                Dropout(self.parameters['dropout']),
                BN(),
                Flatten(),
                Dense(self.parameters['layer']),
                BN(),
                Dropout(self.parameters['dropout']),]

class GaussianGumbelAE(GumbelAE):
    def __init__(self,path,parameters={}):
        if 'G' not in parameters:
            parameters['G'] = 10
        super().__init__(path,parameters)

    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N, G = self.parameters['M'], self.parameters['N'], self.parameters['G'], 
        x = Input(shape=input_shape)
        x_flat = Flatten()(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)
        gumbel = GumbelSoftmax(N,M,
                               self.min_temperature,
                               self.max_temperature,
                               self.anneal_rate)
        gauss  = GaussianSample(G)
        z_cat   = gumbel(pre_encoded)
        z_gauss = gauss (pre_encoded)
        z_cat_flat = Flatten()(z_cat)
        z = merge([z_cat_flat, z_gauss], mode='concat')
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z)

        z_gzero = Input(shape=(G,))
        z2_cat  = Input(shape=(N,M))
        z2_cat_flat = Flatten()(z2_cat)
        z2 = Lambda(K.concatenate)([z2_cat_flat, z_gzero])
        y2 = Sequential(_decoder)(z2)
        
        def loss(x, y):
            kl_loss = gumbel.loss() + gauss.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gumbel.tau)
        self.loss = loss
        self.net         = Model(x, y)
        self.encoder     = Model(x, z_cat)
        self.decoder     = Model([z2_cat,z_gzero],y2)
        self.autoencoder = self.net
        
    def decode(self,data,**kwargs):
        self.load()
        return self.decoder.predict([
            data,
            np.zeros((data.shape[0],self.parameters['G']))],**kwargs)

class GaussianGumbelAE2(GumbelAE2):
    def __init__(self,path,parameters={}):
        if 'G' not in parameters:
            parameters['G'] = 10
        super().__init__(path,parameters)

    def _build(self,input_shape):
        data_dim = np.prod(input_shape)
        print("input_shape:{}, flattened into {}".format(input_shape,data_dim))
        M, N, G = self.parameters['M'], self.parameters['N'], self.parameters['G'], 
        x = Input(shape=input_shape)
        x_flat = Flatten()(x)
        pre_encoded = Sequential(self.build_encoder(input_shape))(x_flat)

        gumbel1 = GumbelSoftmax(N,M,        self.min_temperature, self.max_temperature, self.anneal_rate)
        gumbel2 = GumbelSoftmax(data_dim,2, self.min_temperature, self.max_temperature, self.anneal_rate)
        gumbel3 = GumbelSoftmax(data_dim,2, self.min_temperature, self.max_temperature, self.anneal_rate)
        gumbel3.layers = gumbel2.layers
        gauss   = GaussianSample(G)
        def take_true(y_cat):
            import tensorflow as tf
            return tf.slice(y_cat,[0,0,0],[-1,-1,1])
        _decoder = self.build_decoder(input_shape)
        
        z_cat      = gumbel1(pre_encoded)
        z_cat_flat = Flatten()(z_cat)
        z_gauss    = gauss  (pre_encoded)
        z_flat     = merge([z_cat_flat, z_gauss], mode='concat')

        y = Sequential([
            *_decoder,
            gumbel2,
            Lambda(take_true),
            Reshape(input_shape)
        ])(z_flat)
            
        z2_gzero = Input(shape=(G,))
        z2_cat   = Input(shape=(N,M))
        z2_cat_flat = Flatten()(z2_cat)
        # z2_flat = merge([z2_cat_flat, z_gzero], mode='concat')
        z2_flat = Lambda(K.concatenate)([z2_cat_flat, z2_gzero])
        y2 = Sequential([
            *_decoder,
            gumbel3,
            Lambda(take_true),
            Reshape(input_shape)
        ])(z2_flat)

        def loss(x, y):
            kl_loss = gumbel1.loss() + gumbel2.loss() + gauss.loss()
            reconstruction_loss = bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                      K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel1.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel2.cool))
        self.callbacks.append(LambdaCallback(on_epoch_end=gumbel3.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gumbel1.tau)
        self.loss = loss
        self.net         = Model(x, y)
        self.encoder     = Model(x, z_cat)
        self.decoder     = Model([z2_cat,z2_gzero],y2)
        self.autoencoder = self.net

    def decode(self,data,**kwargs):
        self.load()
        return self.decoder.predict([
            data,
            np.zeros((data.shape[0],self.parameters['G']))],**kwargs)
        
class GaussianConvolutionalGumbelAE(GaussianGumbelAE,ConvolutionalGumbelAE):
    pass

class Discriminator(Network):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
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
        x = Input(shape=input_shape)
        logits = Sequential(self.build_encoder(input_shape))(x)
        gs = GumbelSoftmax(self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(logits)
        z3 = Lambda(lambda z:K.round(z))(z)

        def loss(x, y):
            kl_loss = gs.loss(logits)
            reconstruction_loss = bce(x,y)
            return reconstruction_loss + kl_loss
        
        self.callbacks.append(LambdaCallback(on_epoch_end=gs.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs.tau)
        self.loss = loss
        self.net = Model(x, z)
        self.net_binary = Model(x, z3)
    def _save(self):
        super()._save()
        self.net.save_weights(self.local("net.h5"))
    def _load(self):
        super()._load()
        self.net.load_weights(self.local("net.h5"))
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
        return self.discriminate(data,**kwargs)[:,0]
    def summary(self,verbose=False):
        self.net.summary()
        return self
    
def wrap(x,y,**kwargs):
    "wrap arbitrary operation"
    return Lambda(lambda x:y,**kwargs)(x)

class ActionDiscriminator(Discriminator):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)
    def _build(self,input_shape):
        # Assume a batch of 2N-bit binary vectors.
        # First N bit is a predecessor, last N bit is a successor.
        x = Input(shape=input_shape)
        N = input_shape[0] // 2
        import tensorflow as tf
        pre_T = tf.slice(x,[0,0],[-1,N],name="pre_t")
        suc_T = tf.slice(x,[0,N],[-1,-1],name="suc_t")
        pre_F, suc_F = 1 - pre_T, 1 - suc_T
        TT, TF = pre_T * suc_T, pre_T * suc_F
        FT, FF = pre_F * suc_T, pre_F * suc_F
        _T, _F = suc_T, suc_F
        EQ = TT + FF
        strips_category = tf.stack((TT,TF,FT,FF,_T,_F,EQ),axis=-1)
        strips_category = K.reshape(strips_category,(-1,N,7,1))
        strips_category = wrap(x,strips_category)
        self._strips = Model(x, strips_category)
        
        valid = self.parameters['valid']
        invalid = self.parameters['invalid']
        a = valid + invalid

        # gs1 = GumbelSoftmax(self.min_temperature,self.max_temperature,self.anneal_rate)
        # vars_bernoulli = gs1(vars_logits) # [-1,N,a,2]
        # vars = K.squeeze(tf.slice(vars_bernoulli,[0,0,0,0],[-1,N,a,1]),3)
        # vars = wrap(vars_bernoulli,vars) # [-1,N,a]
        # # note: gumbel-softmax is softmax as well, so it returns a probability too.
        # self.callbacks.append(LambdaCallback(on_epoch_end=gs1.cool))
        # self.custom_log_functions['tau'] = lambda: K.get_value(gs1.tau)
        # def loss_gum(x, y):
        #     return gs1.loss(vars_logits)

        score = Sequential([
            LocallyConnected2D(a,1,7),
            Reshape((N,a))]
        )(strips_category)
        variables = Activation("sigmoid")(score)
        self._variables = Model(x, variables)
        
        def prod1(x,axis):
            l = K.ndim(x)
            origin = [ 0 for i in range(l) ]
            slice = [ -1 for i in range(l) ]
            slice[axis] = 1
            return tf.squeeze(tf.slice(tf.cumprod(x,axis=axis,reverse=True),origin,slice),[axis])
        actions = prod1(variables,1)
        actions = wrap(variables,actions) # [-1, a]
        self._actions = Model(x, actions)

        valid_actions   = tf.slice(actions,[0,0],    [-1,valid])
        invalid_actions = tf.slice(actions,[0,valid],[-1,invalid])
        actions_any = K.sum(valid_actions,axis=1,keepdims=True)
        actions_any = wrap(actions,actions_any)
        self._actions_any = Model(x, actions_any)

        self.loss = bce
        self.net = Model(x, actions_any)
        
    def _save(self):
        super()._save()
        self.net.save_weights(self.local("net.h5"))
    def _load(self):
        super()._load()
        self.net.load_weights(self.local("net.h5"))
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
        return self
    def strips(self,data,**kwargs):
        self.load()
        return self._strips.predict(data,**kwargs)
    def variables(self,data,**kwargs):
        self.load()
        return self._variables.predict(data,**kwargs)
    def actions(self,data,**kwargs):
        self.load()
        return self._actions.predict(data,**kwargs)
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

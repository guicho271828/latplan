#!/usr/bin/env python3

import numpy as np
from functools import reduce
from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, SpatialDropout1D, Lambda, GaussianNoise, LocallyConnected2D
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
        s = ""
        for k in self.custom_log_functions:
            s += "{}: {:5.3g}, ".format(k,self.custom_log_functions[k]())
        for k in logs:
            s += "{}: {:5.3g}, ".format(k,logs[k])
        self.bar.update(epoch, stat=s)
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
    def __init__(self,min,max,anneal_rate):
        self.min = min
        self.anneal_rate = anneal_rate
        self.tau = K.variable(max, name="temperature")
    def call(self,logits):
        u = K.random_uniform(K.shape(logits), 0, 1)
        gumbel = - K.log(-K.log(u + 1e-20) + 1e-20)
        return K.softmax( ( logits + gumbel ) / self.tau )
    def __call__(self,logits):
        return Lambda(self.call)(logits)
    def loss(self, logits):
        q = K.softmax(logits)
        log_q = K.log(q + 1e-20)
        return - K.sum(q * (log_q - K.log(1.0/K.int_shape(logits)[-1])),
                       axis=tuple(range(1,len(K.int_shape(logits)))))
    def cool(self, epoch, logs):
        K.set_value(
            self.tau,
            np.max([self.min,
                    K.get_value(self.tau) * np.exp(- self.anneal_rate * epoch)]))

class GumbelAE(Network):
    def __init__(self,path,M=2,N=25,parameters={}):
        super().__init__(path,{'M':M,'N':N,**parameters})
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
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
        x = Input(shape=input_shape)
        logits = Sequential(self.build_encoder(input_shape))(x)
        gs = GumbelSoftmax(self.min_temperature,self.max_temperature,self.anneal_rate)
        z = gs(logits)
        _decoder = self.build_decoder(input_shape)
        y  = Sequential(_decoder)(z)
        z2 = Input(shape=(N,M))
        y2 = Sequential(_decoder)(z2)
        z3 = Lambda(lambda z:K.round(z))(z)
        y3 = Sequential(_decoder)(z3)

        def loss(x, y):
            kl_loss = gs.loss(logits)
            reconstruction_loss = data_dim * bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                                 K.reshape(y,(K.shape(x)[0],data_dim,)))
            return reconstruction_loss + kl_loss

        self.callbacks.append(LambdaCallback(on_epoch_end=gs.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs.tau)
        self.loss = loss
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
            reconstruction_loss = data_dim * bce(x,y)
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
    
def wrap(x,y):
    "wrap arbitrary operation"
    return Lambda(lambda x:y)(x)

class ActionDiscriminator(Discriminator):
    def __init__(self,path,parameters={}):
        super().__init__(path,parameters)
    def _build(self,input_shape):
        # Assume a batch of 2N-bit binary vectors.
        # First N bit is a predecessor, last N bit is a successor.
        print(input_shape)
        assert len(input_shape) == 1
        # 3rd version: 7 cases
        x = Input(shape=input_shape)
        N = input_shape[0] // 2
        print(N)
        import tensorflow as tf
        print(K.int_shape(x))
        pre_T = tf.slice(x,[0,0],[-1,N],name="pre_t")
        suc_T = tf.slice(x,[0,N],[-1,-1],name="suc_t")
        pre_F = 1 - pre_T
        suc_F = 1 - suc_T
        TT = pre_T * suc_T
        TF = pre_T * suc_F
        FT = pre_F * suc_T
        FF = pre_F * suc_F
        _T = suc_T
        _F = suc_F
        EQ = TT + FF
        strips_category = tf.stack((TT,TF,FT,FF,_T,_F,EQ),axis=-1)
        strips_category = K.reshape(strips_category,(-1,N,7,1))
        strips_category = wrap(x,strips_category)
        a = self.parameters['actions'] + 1 # for none-onf-the-above action
        var_match_logits = Sequential([
            LocallyConnected2D(2*a,1,7), # [-1,N,1,2a]
            Reshape((N,a,2))             # [-1,N,a,2]
        ])(strips_category)
        gs1 = GumbelSoftmax(self.min_temperature,self.max_temperature,self.anneal_rate)
        var_match_bernoulli = gs1(var_match_logits) # [-1,N,a,2]
        var_match = K.squeeze(tf.slice(var_match_bernoulli,[0,0,0,0],[-1,N,a,1]),3) # [-1,N,a]
        var_match = wrap(var_match_bernoulli,var_match)
        # note: gumbel-softmax is softmax as well, so it returns a probability too.
        
        def prod1(tensor,axis):
            # K.prod calls reduce_prod which is not supported on GPU on tensorflow
            # so we cannot use it.
            # action_match_logits = K.prod(var_match,axis=1) # [-1,a]
            l = K.ndim(x)+1
            origin = [ 0 for i in range(l) ]
            slice = [ -1 for i in range(l) ]
            slice[axis] = 1
            return tf.squeeze(tf.slice(tf.cumprod(var_match,axis=axis),origin,slice),[axis])
        action_match = prod1(var_match,1)
        action_match = wrap(var_match_bernoulli,action_match)
        action_unmatch = tf.slice(action_match,[0,a-1],[-1,1])
        action_match_any = 1 - action_unmatch
        action_match_any = wrap(action_match,action_match_any)
        def loss(x, y):
            return bce(x,y) - gs1.loss(var_match_logits)
        def loss_bce(x, y):
            return bce(x,y)
        
        self.callbacks.append(LambdaCallback(on_epoch_end=gs1.cool))
        self.custom_log_functions['tau'] = lambda: K.get_value(gs1.tau)
        self.loss = [loss, loss_bce]
        self.net = Model(x, [action_match_any,action_match_any])
        self._precondition_match_var = Model(x, var_match)
        self._action = Model(x, action_match)
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
    def discriminate(self,data,**kwargs):
        self.load()
        return self.net.predict(data,**kwargs)
    def action(self,data,**kwargs):
        self.load()
        return self._action.predict(data,**kwargs)
    def precondition_match_var(self,data,**kwargs):
        self.load()
        return self._precondition_match_var.predict(data,**kwargs)
    def summary(self,verbose=False):
        self.net.summary()
        return self
    

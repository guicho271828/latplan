#!/usr/bin/env python3

"""
Networks named like XXX2 uses gumbel softmax for the output layer too,
assuming the input/output image is binarized
"""

import numpy as np
from keras.layers import *
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
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from .util.noise import gaussian, salt, pepper
from .util.distances import *
from .util.layers    import *


# utilities ###############################################################

def get(name):
    return globals()[name]

class Network:
    """Base class for various neural networks including GANs, AEs and Classifiers.
Provides an interface for saving / loading the trained weights as well as hyperparameters.

Each instance corresponds to a directory (specified in the `path` variable in the initialization),
which contains the learned weights as well as several json files for the metadata.
If a network depends on another network (i.e. AAE depends on SAE), it is customary to
save the network inside the dependent network. (e.g. Saving an AAE to mnist_SAE/_AAE)

PARAMETERS dict in the initialization argument is stored in the instance as well as 
serialized into a JSON string and is subsequently reloaded along with the weights.
This dict can be used while building the network, making it easier to perform a hyperparameter tuning.
"""
    def __init__(self,path,parameters={}):
        import subprocess
        subprocess.call(["mkdir","-p",path])
        self.path = path
        self.built = False
        self.loaded = False
        self.verbose = True
        self.parameters = parameters
        if "full_epoch" not in parameters:
            if "epoch" in self.parameters:
                # in test time, epoch may not be set
                self.parameters["full_epoch"] = self.parameters["epoch"]
        
        self.custom_log_functions = {}
        self.metrics = []
        import datetime
        self.bar_status_message = ""
        self.bar_shift = 0
        self.bar_epoch = 0
        self.callbacks = [LambdaCallback(on_batch_end=self.bar_update_batch,
                                         on_epoch_end=self.bar_update,
                                         # on_epoch_begin=self.bar_update
                                         ),
                          keras.callbacks.TensorBoard(log_dir=self.local('logs/{}-{}'.format(path,datetime.datetime.now().isoformat())), write_graph=False)]
        
    def build(self,input_shape):
        """An interface for building a network. Input-shape: list of dimensions.
Users should not overload this method; Define _build() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built:
            if self.verbose:
                print("Avoided building {} twice.".format(self))
            return
        self._build(input_shape)
        self.built = True
        if not hasattr(self,"eval"):
            self.eval = self.loss
        return self
    
    def _build(self):
        """An interface for building a network.
This function is called by build() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        pass
    
    def local(self,path):
        """A convenient method for converting a relative path to the learned result directory
into a full path."""
        import os.path as p
        return p.join(self.path,path)
    
    def save(self):
        """An interface for saving a network.
Users should not overload this method; Define _save() for each subclass instead.
This function calls _save bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        print("Saving to {}".format(self.local('')))
        self._save()
        return self
    
    def _save(self):
        """An interface for saving a network.
Users may define a method for each subclass for adding a new save-time feature.
Each method should call the _save() method of the superclass in turn.
Users are not expected to call this method directly. Call save() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        import json
        with open(self.local('aux.json'), 'w') as f:
            json.dump({"parameters":self.parameters,
                       "input_shape":self.net.input_shape[1:]}, f)

    def save_epoch(self, freq=10):
        def fn(epoch, logs):
            if (epoch % freq) == 0:
                self.save()
        return fn
            
    def load(self,allow_failure=False):
        """An interface for loading a network.
Users should not overload this method; Define _load() for each subclass instead.
This function calls _load bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
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
        """An interface for loading a network.
Users may define a method for each subclass for adding a new load-time feature.
Each method should call the _load() method of the superclass in turn.
Users are not expected to call this method directly. Call load() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        import json
        with open(self.local('aux.json'), 'r') as f:
            data = json.load(f)
            self.parameters = data["parameters"]
            self.build(tuple(data["input_shape"]))
        self.net.compile(Adam(0.0001),bce)
        

    def initialize_bar(self):
        import progressbar
        widgets = [
            progressbar.Timer(format='%(elapsed)s'),
            ' ', progressbar.Counter(), 
            progressbar.Bar(),
            progressbar.AbsoluteETA(format='%(eta)s'), ' ',
            DynamicMessage("status")
        ]
        self.bar = progressbar.ProgressBar(max_value=self.max_epoch, widgets=widgets)
        

    msgwidth = 60
    def bar_update_batch(self, batch, logs):
        if not hasattr(self,'bar'):
            self.initialize_bar()

        msg = self.bar_status_message
        
        self.bar_shift = self.bar_shift+1
        if len(msg) > Network.msgwidth:
            b = self.bar_shift % len(msg)
        else:
            b = 0

        self.bar.update(self.bar_epoch,
                        status = "   ".join([msg,msg])[b:b+Network.msgwidth])

    def bar_update(self, epoch, logs):
        "Used for updating the progress bar."
        self.bar_epoch = epoch+1
        
        if not hasattr(self,'bar'):
            self.initialize_bar()
        
        ologs = {}
        for k in self.custom_log_functions:
            ologs[k] = self.custom_log_functions[k]()
        for k in logs:
            ologs[k] = logs[k]
        self.bar_status_message = "  ".join(["{}: {:6.3g}".format(key,value) for key, value in sorted(ologs.items())])
        
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer='adam',lr=0.0001,test_data=None,save=True,report=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
        """Main method for training.
 This method may be overloaded by the subclass into a specific training method, e.g. GAN training."""
        o = getattr(keras.optimizers,optimizer)(lr)
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to

        self.build(train_data.shape[1:])
        if debug:
            self.summary()
        print("parameters",self.parameters)
        print("train_shape",train_data.shape, "test_shape",test_data.shape)

        if isinstance(self.loss,list):
            train_data_to = [ train_data_to for l in self.loss ]
            test_data_to = [ test_data_to for l in self.loss ]
        validation = (test_data,test_data_to) if test_data is not None else None
        try:
            self.max_epoch = epoch
            self.net.compile(optimizer=o, loss=self.loss, metrics=self.metrics)
            self.net.fit(
                train_data, train_data_to,
                epochs=epoch, batch_size=batch_size,
                shuffle=True, validation_data=validation, verbose=False,
                callbacks=self.callbacks)
        except KeyboardInterrupt:
            print("learning stopped\n")
        finally:
            self.net.compile(optimizer=o, loss=self.eval)
        self.loaded = True
        if report:
            self.report(train_data,
                        batch_size=batch_size,
                        test_data=test_data,
                        train_data_to=train_data_to,
                        test_data_to=test_data_to)
        if save:
            self.save()
        return self
    
    def report(self,train_data,
               batch_size=1000,
               test_data=None,
               train_data_to=None,
               test_data_to=None):
        pass

# Network mixins ################################################################

def reg(query, data, d={}):
    if len(query) == 1:
        d[query[0]] = data
        return d
    if query[0] not in d:
        d[query[0]] = {}
    reg(query[1:],data,d[query[0]])
    return d

class AE(Network):
    """Autoencoder class. Supports SAVE and LOAD, as well as REPORT methods.
Additionally, provides ENCODE / DECODE / AUTOENCODE / AUTODECODE methods.
The latter two are used for verifying the performance of the AE.
"""
    def _save(self):
        super()._save()
        self.encoder.save_weights(self.local("encoder.h5"))
        self.decoder.save_weights(self.local("decoder.h5"))
        
    def _load(self):
        super()._load()
        self.encoder.load_weights(self.local("encoder.h5"))
        self.decoder.load_weights(self.local("decoder.h5"))

    def report(self,train_data,
               test_data=None,
               train_data_to=None,
               test_data_to=None,
               batch_size=1000,
               **kwargs):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {'verbose':0,'batch_size':batch_size}

        performance = {}
            
        def test_both(query, fn):
            result = fn(train_data)
            reg(query+["train"], result, performance)
            print(*query,"train", result)
            if test_data is not None:
                result = fn(test_data)
                reg(query+["test"], result, performance)
                print(*query,"test", result)
        
        self.autoencoder.compile(optimizer='adam', loss=self.eval)
        test_both([self.eval.__name__,"vanilla"],
                  lambda data: float(self.autoencoder.evaluate(data,data,**opts)))
        test_both([self.eval.__name__,"gaussian"],
                  lambda data: float(self.autoencoder.evaluate(gaussian(data),data,**opts)))
        test_both([self.eval.__name__,"salt"],
                  lambda data: float(self.autoencoder.evaluate(salt(data),data,**opts)))
        test_both([self.eval.__name__,"pepper"],
                  lambda data: float(self.autoencoder.evaluate(pepper(data),data,**opts)))

        test_both(["activation"],
                  lambda data: float(self.encode_binary(data,batch_size=batch_size,).mean()))
        test_both(["inactive","false"],
                  lambda data: float(self.parameters['N']-np.sum(np.amax(self.encode_binary(data,batch_size=batch_size,),axis=0))))
        test_both(["inactive","true"],
                  lambda data: float(self.parameters['N']-np.sum(np.amax(1-self.encode_binary(data,batch_size=batch_size,),axis=0))))
        test_both(["inactive","both"],
                  lambda data: float(2*self.parameters['N']-np.sum(np.amax(1-self.encode_binary(data,batch_size=batch_size,),axis=0)) -np.sum(np.amax(self.encode_binary(data,batch_size=batch_size,),axis=0))))

        def latent_variance_noise(data,noise):
            encoded = [self.encode_binary(noise(data),batch_size=batch_size,).round() for i in range(10)]
            var = np.var(encoded,axis=0)
            return np.array([np.amax(var), np.amin(var), np.mean(var), np.median(var)]).tolist()
            
        test_both(["variance","vanilla" ], lambda data: latent_variance_noise(data,(lambda x: x)))
        test_both(["variance","gaussian"], lambda data: latent_variance_noise(data,gaussian))
        test_both(["variance","salt"    ], lambda data: latent_variance_noise(data,salt))
        test_both(["variance","pepper"  ], lambda data: latent_variance_noise(data,pepper))

        import json
        with open(self.local('performance.json'), 'w') as f:
            json.dump(performance, f)
        
        import json
        with open(self.local('parameter_count.json'), 'w') as f:
            json.dump(count_params(self.autoencoder), f)
        
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

    def autodecode(self,data,**kwargs):
        self.load()
        return self.autodecoder.predict(data,**kwargs)

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
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                # Entropy Regularization
                alpha = -1.)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions['tau'] = lambda: K.get_value(gs.variable)
            return gs
            
        return fn(**kwargs)

class GumbelAE(AE):
    """An AE whose latent layer is GumbelSofmax.
Fully connected layers only, no convolutions.
Note: references to self.parameters[key] are all hyperparameters."""
    def build_encoder(self,input_shape):
        gs = self.build_gs()
        return [flatten,
                GaussianNoise(self.parameters['noise']),
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
                Dense(self.parameters['N']*self.parameters['M']),
                gs]
    
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            flatten,
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
        _encoder = self.build_encoder(input_shape)
        _decoder = self.build_decoder(input_shape)

        x = Input(shape=input_shape)
        z = Sequential(_encoder)(x)
        y = Sequential(_decoder)(z)
         
        z2 = Input(shape=(self.parameters['N'], self.parameters['M']))
        y2 = Sequential(_decoder)(z2)
        w2 = Sequential(_encoder)(y2)

        def activation(x, y):
            return K.mean(z[:,:,0])
        
        self.loss = BCE
        self.eval = MSE
        self.metrics.append(BCE)
        self.metrics.append(activation)
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.autoencoder = Model(x, y)
        self.autodecoder = Model(z2, w2)
        self.net = self.autoencoder
        y2_downsample = Sequential([
            Reshape((*input_shape,1)),
            MaxPooling2D((2,2))
            ])(y2)
        shape = K.int_shape(y2_downsample)[1:3]
        self.decoder_downsample = Model(z2, Reshape(shape)(y2_downsample))
        self.features = Model(x, Sequential([flatten, *_encoder[:-2]])(x))
        self.custom_log_functions['lr'] = lambda: K.get_value(self.net.optimizer.lr)
        
    def encode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.encode(data,**kwargs)[:,:,0].reshape(-1, N)
    
    def decode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.decode(np.stack((data,1-data),axis=-1),**kwargs)

    def autodecode_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.autodecode(np.stack((data,1-data),axis=-1),**kwargs)[:,:,0].reshape(-1, N)

    def decode_downsample(self,data,**kwargs):
        self.load()
        return self.decoder_downsample.predict(data,**kwargs)

    def decode_downsample_binary(self,data,**kwargs):
        M, N = self.parameters['M'], self.parameters['N']
        assert M == 2, "M={}, not 2".format(M)
        return self.decode_downsample(np.stack((data,1-data),axis=-1),**kwargs)

    def get_features(self, data, **kwargs):
        return self.features.predict(data, **kwargs)

    def plot(self,data,path,verbose=False):
        self.load()
        x = data
        z = self.encode_binary(x)
        y = self.decode_binary(z)
        b = np.round(z)
        by = self.decode_binary(b)

        xg = gaussian(x)
        xs = salt(x)
        xp = pepper(x)

        yg = self.autoencode(xg)
        ys = self.autoencode(xs)
        yp = self.autoencode(xp)

        dy  = ( y-x+1)/2
        dby = (by-x+1)/2
        dyg = (yg-x+1)/2
        dys = (ys-x+1)/2
        dyp = (yp-x+1)/2
        
        from .util.plot import plot_grid, squarify
        _z = squarify(z)
        _b = squarify(b)
        
        images = []
        from .util.plot import plot_grid
        for seq in zip(x, _z, y, dy, _b, by, dby, xg, yg, dyg, xs, ys, dys, xp, yp, dyp):
            images.extend(seq)
        plot_grid(images, w=16, path=path, verbose=verbose)
        return x,z,y,b,by

    def plot_autodecode(self,data,path,verbose=False):
        self.load()
        z = data
        x = self.decode_binary(z)
        
        z2 = self.encode_binary(x)
        z2r = z2.round()
        x2 = self.decode_binary(z2)
        x2r = self.decode_binary(z2r)

        z3 = self.encode_binary(x2)
        z3r = z3.round()
        x3 = self.decode_binary(z3)
        x3r = self.decode_binary(z3r)
        
        M, N = self.parameters['M'], self.parameters['N']

        from .util.plot import plot_grid, squarify
        _z   = squarify(z)
        _z2  = squarify(z2)
        _z2r = squarify(z2r)
        _z3  = squarify(z3)
        _z3r = squarify(z3r)
        
        images = []
        from .util.plot import plot_grid
        for seq in zip(_z, x, _z2, _z2r, x2, x2r, _z3, _z3r, x3, x3r):
            images.extend(seq)
        plot_grid(images, w=10, path=path, verbose=verbose)
        return _z, x, _z2, _z2r

    def plot_variance(self,data,path,verbose=False):
        self.load()
        x = data
        samples = 100
        z = np.array([ np.round(self.encode_binary(x)) for i in range(samples)])
        z = np.einsum("sbz->bsz",z)
        from .util.plot import plot_grid
        plot_grid(z, w=6, path=path, verbose=verbose)

class ConvolutionalGumbelAE(GumbelAE):
    """A mixin that uses convolutions in the encoder."""
    def build_encoder(self,input_shape):
        gs = self.build_gs()
        return [Reshape((*input_shape,1)),
                GaussianNoise(self.parameters['noise']),
                BN(),
                *[Convolution2D(self.parameters['clayer'],(3,3),
                                activation=self.parameters['activation'],padding='same', use_bias=False),
                  Dropout(self.parameters['dropout']),
                  BN(),
                  MaxPooling2D((2,2)),],
                *[Convolution2D(self.parameters['clayer'],(3,3),
                                activation=self.parameters['activation'],padding='same', use_bias=False),
                  Dropout(self.parameters['dropout']),
                  BN(),
                  MaxPooling2D((2,2)),],
                flatten,
                Sequential([
                    Dense(self.parameters['layer'], activation=self.parameters['activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),
                    Dense(self.parameters['N']*self.parameters['M']),
                ]),
                gs]

class Convolutional2GumbelAE(ConvolutionalGumbelAE):
    """A mixin that uses convolutions also in the decoder. Somehow it does not converge."""
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
                *[Dense(self.parameters['layer'], activation='relu', use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                *[Dense(np.prod(last_convolution) * self.parameters['clayer'], activation='relu', use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                Reshape((*last_convolution, self.parameters['clayer'])),
                *[UpSampling2D((2,2)),
                  Deconvolution2D(self.parameters['clayer'],(3,3), activation='relu',padding='same', use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                *[UpSampling2D((2,2)),
                  Deconvolution2D(1,(3,3), activation='sigmoid',padding='same'),],
                Cropping2D(crop),
                Reshape(input_shape),]

# state/action discriminator ####################################################
class Discriminator(Network):
    """Base class for generic binary classifiers."""
    def _build(self,input_shape):
        x = Input(shape=input_shape)
        N = input_shape[0] // 2

        y = Sequential([
            flatten,
            *[Sequential([BN(),
                          Dense(self.parameters['layer'],activation=self.parameters['activation']),
                          Dropout(self.parameters['dropout']),])
              for i in range(self.parameters['num_layers']) ],
            Dense(1,activation="sigmoid")
        ])(x)

        self.loss = bce
        self.net = Model(x, y)
        self.callbacks.append(GradientEarlyStopping(verbose=1,epoch=200,min_grad=self.parameters['min_grad']))
        # self.custom_log_functions['lr'] = lambda: K.get_value(self.net.optimizer.lr)
        
        
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
        self.net.compile(optimizer=optimizer, loss=bce)
        test_both("BCE: {}",
                  lambda data, data_to: self.net.evaluate(data,data_to,**opts))
        return self
    def discriminate(self,data,**kwargs):
        self.load()
        return self.net.predict(data,**kwargs)
    def summary(self,verbose=False):
        self.net.summary()
        return self

class PUDiscriminator(Discriminator):
    """Subclass for PU-learning."""
    def _load(self):
        super()._load()
        K.set_value(self.c, self.parameters['c'])

    def _build(self,input_shape):
        super()._build(input_shape)
        c = K.variable(0, name="c")
        self.c = c
        
        x = Input(shape=input_shape)
        s = self.net(x)
        y2 = wrap(s, s / c)
        self.pu = Model(x,y2)
    
    def discriminate(self,data,**kwargs):
        self.load()
        return self.pu.predict(data,**kwargs)
    
    def train(self,train_data,
              batch_size=1000,
              save=True,
              train_data_to=None,
              test_data=None,
              test_data_to=None,
              **kwargs):
        super().train(train_data,
                      batch_size=batch_size,
                      train_data_to=train_data_to,
                      test_data=test_data,
                      test_data_to=test_data_to,
                      save=False,
                      **kwargs)
        
        s = self.net.predict(test_data[test_data_to == 1],batch_size=batch_size)
        if np.count_nonzero(test_data_to == 1) > 0:
            c = s.mean()
            print("PU constant c =", c)
            K.set_value(self.c, c)
            self.parameters['c'] = float(c)
            # prevent saving before setting c
            if save:
                self.save()
        else:
            raise Exception("there are no positive data in the validation set; Training failed.")
    
class SimpleCAE(AE):
    """A Hack"""
    def build_encoder(self,input_shape):
        return [Reshape((*input_shape,1)),
                GaussianNoise(0.1),
                BN(),
                Convolution2D(self.parameters['clayer'],(3,3),
                              activation=self.parameters['activation'],padding='same', use_bias=False),
                Dropout(self.parameters['dropout']),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters['clayer'],(3,3),
                              activation=self.parameters['activation'],padding='same', use_bias=False),
                Dropout(self.parameters['dropout']),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters['clayer'],(3,3),
                              activation=self.parameters['activation'],padding='same', use_bias=False),
                Dropout(self.parameters['dropout']),
                BN(),
                MaxPooling2D((2,2)),
                flatten,]
    
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(self.parameters['layer'], activation='relu', use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
            Dense(data_dim, activation='sigmoid'),
            Reshape(input_shape),]

    def _build(self,input_shape):
        _encoder = self.build_encoder(input_shape)
        _decoder = self.build_decoder(input_shape)
        
        x = Input(shape=input_shape)
        z = Sequential([flatten, *_encoder])(x)
        y = Sequential(_decoder)(flatten(z))

        z2 = Input(shape=K.int_shape(z)[1:])
        y2 = Sequential(_decoder)(flatten(z2))
        
        self.loss = bce
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.net = Model(x, y)
        self.autoencoder = self.net

    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        pass

def combined_discriminate(data,sae,cae,discriminator,**kwargs):
    images = sae.decode_binary(data,**kwargs)
    data2  = cae.encode(images,**kwargs)
    return discriminator.discriminate(data2,**kwargs)

def combined_discriminate2(data,sae,discriminator,**kwargs):
    _data        = Input(shape=data.shape[1:])
    _data2       = Reshape((*data.shape[1:],1))(_data)
    _categorical = wrap(_data,K.concatenate([_data2, 1-_data2],-1),name="categorical")
    _images      = sae.decoder(_categorical)
    _features    = sae.features(_images)
    _results     = discriminator.net(_features)
    m            = Model(_data, _results)
    return m.predict(data,**kwargs)

class CombinedDiscriminator:
    def __init__(self,sae,cae,discriminator):
        _data        = Input(shape=(sae.parameters['N'],))
        _data2       = Reshape((sae.parameters['N'],1))(_data)
        _categorical = wrap(_data,K.concatenate([_data2, 1-_data2],-1),name="categorical")
        _images      = sae.decoder(_categorical)
        _features    = cae.encoder(_images)
        _results     = discriminator.net(_features)
        m            = Model(_data, _results)
        self.model = m
    
    def __call__(self,data,**kwargs):
        return self.model.predict(data,**kwargs)

class CombinedDiscriminator2(CombinedDiscriminator):
    def __init__(self,sae,discriminator):
        _data        = Input(shape=(sae.parameters['N'],))
        _data2       = Reshape((sae.parameters['N'],1))(_data)
        _categorical = wrap(_data,K.concatenate([_data2, 1-_data2],-1),name="categorical")
        _images      = sae.decoder(_categorical)
        _features    = sae.features(_images)
        _results     = discriminator.net(_features)
        m            = Model(_data, _results)
        self.model = m

# action autoencoder ################################################################

class ActionAE(AE):
    """A network which autoencodes the difference information.

State transitions are not a 1-to-1 mapping in a sense that
there are multiple applicable actions. So you cannot train a newtork that directly learns
a transition S -> T .

We also do not have action labels, so we need to cluster the actions in an unsupervised manner.

This network trains a bidirectional mapping of (S,T) -> (S,A) -> (S,T), given that 
a state transition is a function conditioned by the before-state s.

It is not useful to learn a normal autoencoder (S,T) -> Z -> (S,T) because we cannot separate the
condition and the action label.

We again use gumbel-softmax for representing A."""
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

        def rec(x, y):
            return bce(K.reshape(x,(K.shape(x)[0],dim*2,)),
                       K.reshape(y,(K.shape(x)[0],dim*2,)))
        
        self.metrics.append(rec)
        self.loss = rec
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
            for seq in zip(x_pre_im, x_suc_im, squarify(np.squeeze(z)), y_pre_im, y_suc_im, y_suc_r_im, squarify(np.squeeze(b)), by_pre_im, by_suc_im, by_suc_r_im):
                images.extend(seq)
            plot_grid(images, w=10, path=self.local(path), verbose=verbose)
        else:
            images = []
            for seq in zip(x_pre, x_suc, squarify(np.squeeze(z)), y_pre, y_suc, y_suc_r, squarify(np.squeeze(b)), by_pre, by_suc, by_suc_r):
                images.extend(seq)
            plot_grid(images, w=10, path=self.local(path), verbose=verbose)
        return x,z,y,b,by

class ActionDiscriminator(Discriminator):
    def _build(self,input_shape):
        num_actions = 128
        N = input_shape[0] - num_actions
        x = Input(shape=input_shape)
        pre    = wrap(x,tf.slice(x, [0,0], [-1,N]),name="pre")
        action = wrap(x,tf.slice(x, [0,N], [-1,num_actions]),name="action")

        ys = []
        for i in range(num_actions):
            _x = Input(shape=(N,))
            _y = Sequential([
                flatten,
                *[Sequential([BN(),
                              Dense(self.parameters['layer'],activation=self.parameters['activation']),
                              Dropout(self.parameters['dropout']),])
              for i in range(self.parameters['num_layers']) ],
                Dense(1,activation="sigmoid")
            ])(_x)
            _m = Model(_x,_y,name="action_"+str(i))
            ys.append(_m(pre))

        ys = Concatenate()(ys)
        y  = Dot(-1)([ys,action])

        self.loss = bce
        self.net = Model(x, y)
        self.callbacks.append(GradientEarlyStopping(verbose=1,epoch=50,min_grad=self.parameters['min_grad']))

class ActionPUDiscriminator(PUDiscriminator,ActionDiscriminator):
    pass

# imbalanced data WIP ###############################################################

# In general, there are more invalid data than valid data. These kinds of
# imbalanced datasets always make it difficult to train a classifier.
# Theoretically, the most promising way for this problem is Undersampling + bagging.
# Yeah I know, I am not a statistician. But I have a true statistician friend !
# TBD : add reference to that paper (I forgot).

# Ultimately this implementation was not used during AAAI submission.

class UBDiscriminator(Discriminator):
    def _build(self,input_shape):
        x = Input(shape=input_shape)

        self.discriminators = []
        for i in range(self.parameters['bagging']):
            d = Discriminator(self.path+"/"+str(i),self.parameters)
            d.build(input_shape)
            self.discriminators.append(d)

        y = average([ d.net(x) for d in self.discriminators ])
        y = wrap(y,K.round(y))
        self.net = Model(x,y)
        self.net.compile(optimizer='adam',loss=bce)
        
    def train(self,train_data,
              train_data_to=None,
              test_data=None,
              test_data_to=None,
              *args,**kwargs):

        self.build(train_data.shape[1:])
        
        num   = len(test_data_to)
        num_p = np.count_nonzero(test_data_to)
        num_n = num-num_p
        assert num_n > num_p
        print("positive : negative = ",num_p,":",num_n,"negative ratio",num_n/num_p)

        ind_p = np.where(test_data_to == 1)[0]
        ind_n = np.where(test_data_to == 0)[0]
        
        from numpy.random import shuffle
        shuffle(ind_n)
        
        per_bag = num_n // len(self.discriminators)
        for i, d in enumerate(self.discriminators):
            print("training",i+1,"/",len(self.discriminators),"th discriminator")
            ind_n_per_bag = ind_n[per_bag*i:per_bag*(i+1)]
            ind = np.concatenate((ind_p,ind_n_per_bag))
            d.train(train_data[ind],
                    train_data_to=train_data_to[ind],
                    test_data=test_data,
                    test_data_to=test_data_to,
                    *args,**kwargs)

    def discriminate(self,data,**kwargs):
        return self.net.predict(data,**kwargs)


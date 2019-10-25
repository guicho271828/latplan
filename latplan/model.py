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
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback, CallbackList
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from .util.noise import gaussian, salt, pepper
from .util.distances import *
from .util.layers    import *
from .util.perminv   import *
from .util.tuning    import InvalidHyperparameterError

# utilities ###############################################################

def get(name):
    return globals()[name]

def get_ae_type(directory):
    import os.path
    import json
    with open(os.path.join(directory,"aux.json"),"r") as f:
        return json.load(f)["class"]

def load(directory,allow_failure=False):
    if allow_failure:
        try:
            classobj = get(get_ae_type(directory))
        except FileNotFoundError as c:
            print("Error skipped:", c)
            return None
    else:
        classobj = get(get_ae_type(directory))
    return classobj(directory).load(allow_failure=allow_failure)

def ensure_list(x):
    if type(x) is not list:
        return [x]
    else:
        return x

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
        self.compiled = False
        self.loaded = False
        self.verbose = True
        self.parameters = parameters
        if "full_epoch" not in parameters:
            if "epoch" in self.parameters:
                # in test time, epoch may not be set
                self.parameters["full_epoch"] = self.parameters["epoch"]
        
        self.custom_log_functions = {}
        self.metrics = []
        self.nets    = [None]
        self.losses  = [None]
        import datetime
        self.callbacks = [LambdaCallback(on_epoch_end=self.bar_update,
                                         # on_epoch_begin=self.bar_update
                                         ),
                          keras.callbacks.TensorBoard(log_dir=self.local('logs/{}-{}'.format(path,datetime.datetime.now().isoformat())), write_graph=False)]
        
    def build(self,*args,**kwargs):
        """An interface for building a network. Input-shape: list of dimensions.
Users should not overload this method; Define _build() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built:
            if self.verbose:
                print("Avoided building {} twice.".format(self))
            return
        self._build(*args,**kwargs)
        self.built = True
        return self
    
    def _build(self,*args,**kwargs):
        """An interface for building a network.
This function is called by build() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        pass
    
    def compile(self,*args,**kwargs):
        """An interface for compiling a network."""
        if self.compiled:
            if self.verbose:
                print("Avoided compiling {} twice.".format(self))
            return
        self._compile(*args,**kwargs)
        self.compiled = True
        return self
    
    def _compile(self,optimizers):
        """An interface for compileing a network."""
        # default method.
        for net, o, loss in zip(self.nets, optimizers, self.losses):
            net.compile(optimizer=o, loss=loss, metrics=self.metrics)
    
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
        for i, net in enumerate(self.nets):
            net.save_weights(self.local("net{}.h5".format(i)))

        import json
        with open(self.local('aux.json'), 'w') as f:
            json.dump({"parameters":self.parameters,
                       "class"     :self.__class__.__name__,
                       "input_shape":self.net.input_shape[1:]}, f , skipkeys=True)

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
        for i, net in enumerate(self.nets):
            net.load_weights(self.local("net{}.h5".format(i)))
        
    def initialize_bar(self):
        import progressbar
        widgets = [
            progressbar.Timer(format='%(elapsed)s'),
            ' ', progressbar.Counter(), ' | ',
            # progressbar.Bar(),
            progressbar.AbsoluteETA(format='%(eta)s'), ' ',
            DynamicMessage("status")
        ]
        self.bar = progressbar.ProgressBar(max_value=self.max_epoch, widgets=widgets)
        
    def bar_update(self, epoch, logs):
        "Used for updating the progress bar."
        
        if not hasattr(self,'bar'):
            self.initialize_bar()
        
        tlogs = {}
        for k in self.custom_log_functions:
            tlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if "val" not in k:
                tlogs[k] = logs[k]
        vlogs = {}
        for k in self.custom_log_functions:
            vlogs[k] = self.custom_log_functions[k]()
        for k in logs:
            if "val" in k:
                vlogs[k[4:]] = logs[k]
        
        if (epoch % 10) == 9:
            self.bar.update(epoch+1, status = "[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())]) + "\n")
        else:
            self.bar.update(epoch+1, status = "[t] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(tlogs.items())]))

    @property
    def net(self):
        return self.nets[0]

    @net.setter
    def net(self,net):
        self.nets[0] = net
        return net
    
    @property
    def loss(self):
        return self.losses[0]

    @loss.setter
    def loss(self,loss):
        self.losses[0] = loss
        return loss
    
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer='adam',lr=0.0001,test_data=None,save=True,report=True,
              train_data_to=None,
              test_data_to=None,
              **kwargs):
        """Main method for training.
 This method may be overloaded by the subclass into a specific training method, e.g. GAN training."""

        if test_data     is None:
            test_data     = train_data
        if train_data_to is None:
            train_data_to = train_data
        if test_data_to  is None:
            test_data_to  = test_data

        self.max_epoch = epoch
        self.build(train_data.shape[1:]) # depends on self.optimizer
        print("parameters",self.parameters)

        def replicate(thing):
            if isinstance(thing, tuple):
                thing = list(thing)
            if isinstance(thing, list):
                assert len(thing) == len(self.nets)
                return thing
            else:
                return [thing for _ in self.nets]
          
        train_data    = replicate(train_data)
        train_data_to = replicate(train_data_to)
        test_data     = replicate(test_data)
        test_data_to  = replicate(test_data_to)
        optimizer     = replicate(optimizer)
        lr            = replicate(lr)
        
        def get_optimizer(optimizer,lr):
            return getattr(keras.optimizers,optimizer)(lr)

        self.compile(list(map(get_optimizer, optimizer, lr)))

        def assert_length(data):
            l = None
            for subdata in data:
                if not ((l is None) or (len(subdata) == l)):
                    return False
                l = len(subdata)
            return True
        
        assert assert_length(train_data   )
        assert assert_length(train_data_to)
        assert assert_length(test_data    )
        assert assert_length(test_data_to )
    
        def make_batch(subdata):
            # len: 15, batch: 5 -> 3 : 19//5 = 3
            # len: 14, batch: 5 -> 3 : 18//5 = 3
            # len: 16, batch: 5 -> 4 : 20//5 = 4
            for i in range((len(subdata)+batch_size-1)//batch_size):
                yield subdata[i*batch_size:min((i+1)*batch_size,len(subdata))]

        index_array = np.arange(len(train_data[0]))
        np.random.shuffle(index_array)
        
        clist = CallbackList(callbacks=self.callbacks)
        clist.set_model(self.nets[0])
        clist.set_params({
            'batch_size': batch_size,
            'epochs': epoch,
            'steps': None,
            'samples': len(train_data[0]),
            'verbose': 0,
            'do_validation': False,
            'metrics': [],
        })

        def generate_logs(data,data_to):
            logs   = {}
            losses = []
            for i, (net, subdata, subdata_to) in enumerate(zip(self.nets, data, data_to)):
                evals = net.evaluate(subdata,
                                     subdata_to,
                                     batch_size=batch_size,
                                     verbose=0)
                logs = { k:v for k,v in zip(net.metrics_names, ensure_list(evals)) }
                losses.append(logs["loss"])
            if len(losses) > 2:
                for i, loss in enumerate(losses):
                    logs["loss"+str(i)] = loss
            logs["loss"] = np.sum(losses)
            return logs
        
        try:
            clist.on_train_begin()
            logs = {}
            indices_cache       = [ indices for indices in make_batch(index_array) ]
            train_data_cache    = [[ train_subdata   [indices] for train_subdata    in train_data    ] for indices in indices_cache ]
            train_data_to_cache = [[ train_subdata_to[indices] for train_subdata_to in train_data_to ] for indices in indices_cache ]
            for epoch in range(epoch):
                clist.on_epoch_begin(epoch,logs)
                for train_subdata_cache,train_subdata_to_cache in zip(train_data_cache,train_data_to_cache):
                    for net,train_subdata_batch_cache,train_subdata_to_batch_cache in zip(self.nets, train_subdata_cache,train_subdata_to_cache):
                        net.train_on_batch(train_subdata_batch_cache, train_subdata_to_batch_cache)

                logs = generate_logs(train_data, train_data_to)
                for k,v in generate_logs(test_data,  test_data_to).items():
                    logs["val_"+k] = v
                clist.on_epoch_end(epoch,logs)
            clist.on_train_end()
        
        except KeyboardInterrupt:
            print("learning stopped\n")
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
    
    def evaluate(self,*args,**kwargs):

        return np.sum([
            { k:v for k,v in zip(net.metrics_names,
                                 ensure_list(net.evaluate(*args,**kwargs)))}["loss"]
            for net in self.nets
        ])

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
    def _report(self,test_both,**opts):

        from .util.np_distances import mse

        test_both(["MSE","vanilla"],
                  lambda data: mse(data, self.autoencode(data,**opts)))
        test_both(["MSE","gaussian"],
                  lambda data: mse(data, self.autoencode(gaussian(data),**opts)))
        test_both(["MSE","salt"],
                  lambda data: mse(data, self.autoencode(salt(data),**opts)))
        test_both(["MSE","pepper"],
                  lambda data: mse(data, self.autoencode(pepper(data),**opts)))

        test_both(["activation"],
                  lambda data: float(self.encode(data,**opts).mean()))
        test_both(["ever_1"],
                  lambda data: float(np.sum(np.amax(self.encode(data,**opts),axis=0))))
        test_both(["ever_0"],
                  lambda data: float(np.sum(1-np.amin(self.encode(data,**opts),axis=0))))
        test_both(["effective"],
                  lambda data: float(np.sum((1-np.amin(self.encode(data,**opts),axis=0))*np.amax(self.encode(data,**opts),axis=0))))

        def latent_variance_noise(data,noise):
            encoded = [self.encode(noise(data),**opts).round() for i in range(10)]
            var = np.var(encoded,axis=0)
            return np.array([np.amax(var), np.amin(var), np.mean(var), np.median(var)]).tolist()
            
        test_both(["variance","vanilla" ], lambda data: latent_variance_noise(data,(lambda x: x)))
        test_both(["variance","gaussian"], lambda data: latent_variance_noise(data,gaussian))
        test_both(["variance","salt"    ], lambda data: latent_variance_noise(data,salt))
        test_both(["variance","pepper"  ], lambda data: latent_variance_noise(data,pepper))
        return

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
        
        self._report(test_both,**opts)

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
               full_epoch=self.parameters['full_epoch'],
               offset=0,
               argmax=self.parameters['argmax'],
               alpha=-1.):
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                offset=offset,
                test_gumbel=not argmax,
                test_softmax=not argmax,
                # Entropy Regularization
                alpha = alpha)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions['tau'] = lambda: K.get_value(gs.variable)
            return gs
            
        return fn(**kwargs)

# Latent Activations ################################################################

class ConcreteLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters['M'] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters['N'],)
    def zindim(self):
        return (self.parameters['N'],self.parameters['M'],)
    def activation(self):
        return Sequential([
            self.build_gs(),
            take_true(),
        ])

class QuantizedLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters['M'] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters['N'],)
    def zindim(self):
        return (self.parameters['N'],)
    def activation(self):
        return heavyside()

class SigmoidLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters['M'] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters['N'],)
    def zindim(self):
        return (self.parameters['N'],)
    def activation(self):
        return rounded_sigmoid()

class GumbelSoftmaxLatentMixin:
    def zdim(self):
        return (self.parameters['N']*self.parameters['M'],)
    def zindim(self):
        return (self.parameters['N']*self.parameters['M'],)
    def activation(self):
        return Sequential([
            self.build_gs(),
            flatten,
        ])

class SoftmaxLatentMixin:
    def zdim(self):
        return (self.parameters['N']*self.parameters['M'],)
    def zindim(self):
        return (self.parameters['N']*self.parameters['M'],)
    def activation(self):
        return Sequential([
            Reshape((self.parameters['N'],self.parameters['M'],)),
            rounded_softmax(),
            flatten,
        ])

# Encoder / Decoder ################################################################

class FullConnectedEncoderMixin:
    def build_encoder(self,input_shape):
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
                Dense(np.prod(self.zindim())),
                self.activation(),
        ]

class FullConnectedDecoderMixin:
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

class ConvolutionalEncoderMixin:
    """A mixin that uses convolutions in the encoder."""
    def build_encoder(self,input_shape):
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
                    Dense(np.prod(self.zindim())),
                ]),
                self.activation(),
        ]

class ConvolutionalDecoderMixin:
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

# State Auto Encoder ################################################################

class GumbelAE(AE):
    """An AE whose latent layer is GumbelSofmax.
Fully connected layers only, no convolutions.
Note: references to self.parameters[key] are all hyperparameters."""
    def build_encoder(self,input_shape):
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
                self.build_gs(),
                take_true(),
        ]
    
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
         
        z2 = Input(shape=K.int_shape(z)[1:])
        y2 = Sequential(_decoder)(z2)
        w2 = Sequential(_encoder)(y2)

        self.loss = BCE
        self.metrics.append(BCE)
        self.metrics.append(MSE)
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.autoencoder = Model(x, y)
        self.autodecoder = Model(z2, w2)
        self.net = self.autoencoder
        self.features = Model(x, Sequential([flatten, *_encoder[:-2]])(x))
        self.custom_log_functions['lr'] = lambda: K.get_value(self.net.optimizer.lr)
        
    def get_features(self, data, **kwargs):
        return self.features.predict(data, **kwargs)

    def plot(self,data,path,verbose=False):
        self.load()
        x = data
        z = self.encode(x)
        y = self.decode(z)
        b = np.round(z)
        by = self.decode(b)

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
        x = self.decode(z)
        
        z2 = self.encode(x)
        z2r = z2.round()
        x2 = self.decode(z2)
        x2r = self.decode(z2r)

        z3 = self.encode(x2)
        z3r = z3.round()
        x3 = self.decode(z3)
        x3r = self.decode(z3r)
        
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
        z = np.array([ np.round(self.encode(x)) for i in range(samples)])
        z = np.einsum("sbz->bsz",z)
        from .util.plot import plot_grid
        plot_grid(z, w=6, path=path, verbose=verbose)

# Mixins ################################################################

class ZeroSuppressMixin:
    def _build(self,input_shape):
        super()._build(input_shape)
        
        alpha = LinearSchedule(schedule={
            0:0,
            (self.parameters["epoch"]//3):0,
            (self.parameters["epoch"]//3)*2:self.parameters["zerosuppress"]
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))
        
        zerosuppress_loss = K.mean(self.encoder.output)

        self.net.add_loss(K.in_train_phase(zerosuppress_loss * alpha.variable, 0.0))
        
        def activation(x, y):
            return zerosuppress_loss
        
        def zerosup_alpha(x, y):
            return alpha.variable
        
        self.metrics.append(activation)
        self.metrics.append(zerosup_alpha)
        return

# The original Gumbel Softmax formulation that minimizes the KL divergence
# between the latent and the the Bernoulli(0.5)
class NGMixin:
    def build_gs(self,
                 **kwargs):

        def fn(N=self.parameters['N'],
               M=self.parameters['M'],
               max_temperature=self.parameters['max_temperature'],
               min_temperature=self.parameters['min_temperature'],
               full_epoch=self.parameters['full_epoch'],
               argmax=self.parameters['argmax'],
               alpha=1.):       # positive alpha
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                test_gumbel=not argmax,
                test_softmax=not argmax,
                # Entropy Regularization
                alpha = alpha)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions['tau'] = lambda: K.get_value(gs.variable)
            return gs
            
        return fn(**kwargs)
    
# The version that does not maximize nor minimize the KL divergence while
# still adding noise
class NoKLMixin:
    def build_gs(self,
                 **kwargs):

        def fn(N=self.parameters['N'],
               M=self.parameters['M'],
               max_temperature=self.parameters['max_temperature'],
               min_temperature=self.parameters['min_temperature'],
               full_epoch=self.parameters['full_epoch'],
               argmax=self.parameters['argmax'],
               alpha=0.):       # positive alpha
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                test_gumbel=not argmax,
                test_softmax=not argmax,
                # Entropy Regularization
                alpha = alpha)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions['tau'] = lambda: K.get_value(gs.variable)
            return gs
            
        return fn(**kwargs)

# The version that does not use Gumbel noise
class DetMixin:
    def build_gs(self,
                 **kwargs):

        def fn(N=self.parameters['N'],
               M=self.parameters['M'],
               max_temperature=self.parameters['max_temperature'],
               min_temperature=self.parameters['min_temperature'],
               full_epoch=self.parameters['full_epoch'],
               argmax=self.parameters['argmax'],
               alpha=0.):       # positive alpha
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                train_gumbel=False,
                test_gumbel=not argmax,
                test_softmax=not argmax,
                # Entropy Regularization
                alpha = alpha)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions['tau'] = lambda: K.get_value(gs.variable)
            return gs
            
        return fn(**kwargs)


# The variant that takes transitions instead of states
class TransitionAE(GumbelAE):
    def double_mode(self):
        self.mode(False)
    def single_mode(self):
        self.mode(True)
    def mode(self, single):
        if single:
            self.encoder     = self.s_encoder     
            self.decoder     = self.s_decoder     
            self.autoencoder = self.s_autoencoder 
            self.autodecoder = self.s_autodecoder
        else:
            self.encoder     = self.d_encoder     
            self.decoder     = self.d_decoder     
            self.autoencoder = self.d_autoencoder 
            self.autodecoder = self.d_autodecoder 

    def as_single(self, fn, data, *args, **kwargs):
        self.single_mode()
        try:
            if data.shape[1] == 2:
                return fn(data[:, 0, ...],*args,**kwargs)
            else:
                return fn(data,*args,**kwargs)
        finally:
            self.double_mode()
        
    def plot(self, data, *args, **kwargs):
        return self.as_single(super().plot, data, *args, **kwargs)
    def plot_autodecode(self, data, *args, **kwargs):
        return self.as_single(super().plot_autodecode, data, *args, **kwargs)
    def plot_variance(self, data, *args, **kwargs):
        return self.as_single(super().plot_variance, data, *args, **kwargs)

    def adaptively(self, fn, data, *args, **kwargs):
        try:
            if data.shape[1] == 2:
                self.double_mode()
                return fn(data,*args,**kwargs)
            else:
                self.single_mode()
                return fn(data,*args,**kwargs)
        finally:
            self.double_mode()

    def encode(self, data, *args, **kwargs):
        return self.adaptively(super().encode, data, *args, **kwargs)
    def decode(self, data, *args, **kwargs):
        return self.adaptively(super().decode, data, *args, **kwargs)
    def autoencode(self, data, *args, **kwargs):
        return self.adaptively(super().autoencode, data, *args, **kwargs)
    def autodecode(self, data, *args, **kwargs):
        return self.adaptively(super().autodecode, data, *args, **kwargs)
    
    def _build(self,input_shape):
        # [batch, 2, ...] -> [batch, ...]
        _encoder = self.build_encoder(input_shape[1:])
        _decoder = self.build_decoder(input_shape[1:])
        self.encoder_net = _encoder
        self.decoder_net = _decoder

        x = Input(shape=input_shape[1:])
        z = Sequential(_encoder)(x)
        y = Sequential(_decoder)(z)
         
        z2 = Input(shape=K.int_shape(z)[1:])
        y2 = Sequential(_decoder)(z2)
        w2 = Sequential(_encoder)(y2)

        self.s_encoder     = Model(x, z)
        self.s_decoder     = Model(z2, y2)
        self.s_autoencoder = Model(x, y)
        self.s_autodecoder = Model(z2, w2)

        x               = Input(shape=input_shape)
        z, z_pre, z_suc = dapply(x, Sequential(_encoder))
        y, _,     _     = dapply(z, Sequential(_decoder))
        
        z2       = Input(shape=K.int_shape(z)[1:])
        y2, _, _ = dapply(z2, Sequential(_decoder))
        w2, _, _ = dapply(y2, Sequential(_encoder))
        

        self.loss = BCE
        self.eval = MSE
        self.metrics.append(BCE)
        self.metrics.append(MSE)
        self.net = Model(x, y)

        self.d_encoder     = Model(x, z)
        self.d_decoder     = Model(z2, y2)
        self.d_autoencoder = Model(x, y)
        self.d_autodecoder = Model(z2, w2)

        self.double_mode()
        return

class HammingLoggerMixin:
    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)
        test_both(["hamming"],
                  lambda data: \
                    float( \
                      np.mean( \
                        abs(self.encode(data[:,0,...]) \
                            - self.encode(data[:,1,...])))))
        return

    def _build(self,input_shape):
        super()._build(input_shape)
        def hamming(x, y):
            return K.mean(mae(self.encoder.output[:,0,...],
                              self.encoder.output[:,1,...]))
        self.metrics.append(hamming)
        return

class LocalityMixin:
    def _build(self,input_shape):
        super()._build(input_shape)
        
        self.locality_alpha = LinearSchedule(schedule={
            self.parameters["locality"]:self.parameters["locality"]
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=self.locality_alpha.update))
        
        # def locality_alpha(x, y):
        #     return self.locality_alpha.variable
        # 
        # self.metrics.append(locality_alpha)
        return

class HammingMixin(LocalityMixin, HammingLoggerMixin):
    def _build(self,input_shape):
        super()._build(input_shape)
        loss = K.mean(mae(self.encoder.output[:,0,...],
                          self.encoder.output[:,1,...]))
        self.net.add_loss(K.in_train_phase(loss * self.locality_alpha.variable, 0.0))
        return

class CosineMixin (LocalityMixin, HammingLoggerMixin):
    def _build(self,input_shape):
        super()._build(input_shape)
        loss = K.mean(keras.losses.cosine_proximity(self.encoder.output[:,0,...],
                                                    self.encoder.output[:,1,...]))
        self.net.add_loss(K.in_train_phase(loss * self.locality_alpha.variable, 0.0))
        def cosine(x, y):
            return loss
        self.metrics.append(cosine)
        return

class PoissonMixin(LocalityMixin, HammingLoggerMixin):
    def _build(self,input_shape):
        super()._build(input_shape)
        loss = K.mean(keras.losses.poisson(self.encoder.output[:,0,...],
                                           self.encoder.output[:,1,...]))
        self.net.add_loss(K.in_train_phase(loss * self.locality_alpha.variable, 0.0))
        def poisson(x, y):
            return loss
        self.metrics.append(poisson)
        return
    
# Zero-sup SAE ###############################################################

class ConvolutionalGumbelAE(ConvolutionalMixin, GumbelAE):
    pass
class Convolutional2GumbelAE(Convolutional2Mixin, GumbelAE):
    pass
class NGGumbelAE(NGMixin, GumbelAE):
    pass
class NGConvolutionalGumbelAE(NGMixin, ConvolutionalMixin, GumbelAE):
    pass
class NGConvolutional2GumbelAE(NGMixin, Convolutional2Mixin, GumbelAE):
    pass
class NoKLGumbelAE(NoKLMixin, GumbelAE):
    pass
class NoKLConvolutionalGumbelAE(NoKLMixin, ConvolutionalMixin, GumbelAE):
    pass
class NoKLConvolutional2GumbelAE(NoKLMixin, Convolutional2Mixin, GumbelAE):
    pass


class ZeroSuppressGumbelAE(ZeroSuppressMixin, GumbelAE):
    pass
class ZeroSuppressConvolutionalGumbelAE(ZeroSuppressMixin, ConvolutionalMixin, GumbelAE):
    pass
class ZeroSuppressConvolutional2GumbelAE(ZeroSuppressMixin, Convolutional2Mixin, GumbelAE):
    pass
class NGZeroSuppressGumbelAE(NGMixin, ZeroSuppressMixin, GumbelAE):
    pass
class NGZeroSuppressConvolutionalGumbelAE(NGMixin, ZeroSuppressMixin, ConvolutionalMixin, GumbelAE):
    pass
class NGZeroSuppressConvolutional2GumbelAE(NGMixin, ZeroSuppressMixin, Convolutional2Mixin, GumbelAE):
    pass
class NoKLZeroSuppressGumbelAE(NoKLMixin, ZeroSuppressMixin, GumbelAE):
    pass
class NoKLZeroSuppressConvolutionalGumbelAE(NoKLMixin, ZeroSuppressMixin, ConvolutionalMixin, GumbelAE):
    pass
class NoKLZeroSuppressConvolutional2GumbelAE(NoKLMixin, ZeroSuppressMixin, Convolutional2Mixin, GumbelAE):
    pass
class DetZeroSuppressGumbelAE(DetMixin, ZeroSuppressMixin, GumbelAE):
    pass
class DetZeroSuppressConvolutionalGumbelAE(DetMixin, ZeroSuppressMixin, ConvolutionalMixin, GumbelAE):
    pass
class DetZeroSuppressConvolutional2GumbelAE(DetMixin, ZeroSuppressMixin, Convolutional2Mixin, GumbelAE):
    pass

# Transition SAE ################################################################

class HammingTransitionAE(HammingMixin, ZeroSuppressMixin, ConvolutionalMixin, TransitionAE, GumbelAE):
    pass
class CosineTransitionAE (CosineMixin,  ZeroSuppressMixin, ConvolutionalMixin, TransitionAE, GumbelAE):
    pass
class PoissonTransitionAE(PoissonMixin, ZeroSuppressMixin, ConvolutionalMixin, TransitionAE, GumbelAE):
    pass

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

_combined = None
def combined_sd(states,sae,cae,sd3,**kwargs):
    global _combined
    if _combined is None:
        x = Input(shape=states.shape[1:])
        tmp = x
        if sd3.parameters["method"] == "direct":
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "feature":
            tmp = sae.decoder(tmp)
            tmp = sae.features(tmp)
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "cae":
            tmp = sae.decoder(tmp)
            tmp = cae.encoder(tmp)
            tmp = sd3.net(tmp)
        if sd3.parameters["method"] == "image":
            tmp = sae.decoder(tmp)
            tmp = cae.encoder(tmp)
            tmp = sd3.net(tmp)
        _combined = Model(x, tmp)
    return _combined.predict(states, **kwargs)

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
            Sequential([
                    Dense(self.parameters['N']*self.parameters['M']),
                    self.build_gs(),
            ]),
        ]
    
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
                Dense(data_dim, activation=Lambda(lambda x: K.in_train_phase(K.sigmoid(x), K.round(K.sigmoid(x))))),
                Reshape(input_shape),]),]
   
    def _build(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters['M'], self.parameters['N']
        
        x = Input(shape=input_shape)
        
        pre = wrap(x,x[:,:dim],name="pre")
        suc = wrap(x,x[:,dim:],name="suc")

        print("encoder")
        _encoder = self.build_encoder([dim])
        action = ConditionalSequential(_encoder, pre, axis=1)(suc)
        
        print("decoder")
        _decoder = self.build_decoder([dim])
        suc_reconstruction = ConditionalSequential(_decoder, pre, axis=1)(flatten(action))
        y = Concatenate(axis=1)([pre,suc_reconstruction])
        
        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        suc_reconstruction2 = ConditionalSequential(_decoder, pre2, axis=1)(flatten(action2))
        y2 = Concatenate(axis=1)([pre2,suc_reconstruction2])

        self.metrics.append(MAE)
        self.loss = BCE
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
    
    def plot(self,data,path,verbose=False,sae=None):
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

        if sae:
            x_pre_im, x_suc_im = sae.decode(x[:,:dim]), sae.decode(x[:,dim:])
            y_pre_im, y_suc_im = sae.decode(y[:,:dim]), sae.decode(y[:,dim:])
            by_pre_im, by_suc_im = sae.decode(by[:,:dim]), sae.decode(by[:,dim:])
            y_suc_r_im, by_suc_r_im = sae.decode(y[:,dim:].round()), sae.decode(by[:,dim:].round())
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


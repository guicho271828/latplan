#!/usr/bin/env python3

"""
Networks named like XXX2 uses gumbel softmax for the output layer too,
assuming the input/output image is binarized
"""

import json
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
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback, CallbackList, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
from .util.noise import gaussian, salt, pepper
from .util.distances import *
from .util.layers    import *
from .util.perminv   import *
from .util.tuning    import InvalidHyperparameterError
from .util           import ensure_list, NpEncoder

# utilities ###############################################################

def get(name):
    return globals()[name]

def get_ae_type(directory):
    import os.path
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
        self.built_aux = False
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

    def build_aux(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
Users should not overload this method; Define _build() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built_aux:
            if self.verbose:
                print("Avoided building {} twice.".format(self))
            return
        self._build_aux(*args,**kwargs)
        self.built_aux = True
        return self

    def _build_aux(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
This function is called by build_aux() only when the network is not build yet.
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

        with open(self.local('aux.json'), 'w') as f:
            json.dump({"parameters":self.parameters,
                       "class"     :self.__class__.__name__,
                       "input_shape":self.net.input_shape[1:]}, f , skipkeys=True, cls=NpEncoder)

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
        with open(self.local('aux.json'), 'r') as f:
            data = json.load(f)
            self.parameters = data["parameters"]
            self.build(tuple(data["input_shape"]))
            self.build_aux(tuple(data["input_shape"]))
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
              epoch=200,batch_size=1000,optimizer='adam',lr=0.0001,val_data=None,save=True,
              train_data_to=None,
              val_data_to=None,
              **kwargs):
        """Main method for training.
 This method may be overloaded by the subclass into a specific training method, e.g. GAN training."""

        if val_data     is None:
            val_data     = train_data
        if train_data_to is None:
            train_data_to = train_data
        if val_data_to  is None:
            val_data_to  = val_data

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
        val_data     = replicate(val_data)
        val_data_to  = replicate(val_data_to)
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
        assert assert_length(val_data    )
        assert assert_length(val_data_to )

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
        self.nets[0].stop_training = False

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
                for k,v in generate_logs(val_data,  val_data_to).items():
                    logs["val_"+k] = v
                clist.on_epoch_end(epoch,logs)
                if self.nets[0].stop_training:
                    break
            clist.on_train_end()

        except KeyboardInterrupt:
            print("learning stopped\n")
        self.loaded = True
        if save:
            self.save()
        self.build_aux(train_data.shape[1:]) # depends on self.optimizer
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

        with open(self.local('performance.json'), 'w') as f:
            json.dump(performance, f, cls=NpEncoder)

        with open(self.local('parameter_count.json'), 'w') as f:
            json.dump(count_params(self.autoencoder), f, cls=NpEncoder)

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
        # python methods cannot use self in the
        # default values, because python sucks

        def fn(N               = self.parameters['N'],
               M               = self.parameters['M'],
               max_temperature = self.parameters['max_temperature'],
               min_temperature = self.parameters['min_temperature'],
               full_epoch      = self.parameters['full_epoch'],
               train_gumbel    = self.parameters['train_gumbel'],
               test_gumbel     = self.parameters['test_gumbel'],
               test_softmax    = self.parameters['test_softmax'],
               beta            = self.parameters['beta'],
               offset          = 0):
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                offset        = offset,
                train_gumbel  = train_gumbel,
                test_gumbel   = test_gumbel,
                test_softmax  = test_softmax,
                beta          = beta)
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
                *[Dense(self.parameters['layer'],
                        activation=self.parameters['activation'],
                        use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                *[Dense(np.prod(last_convolution) * self.parameters['clayer'],
                        activation=self.parameters['activation'],
                        use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                Reshape((*last_convolution, self.parameters['clayer'])),
                *[UpSampling2D((2,2)),
                  Deconvolution2D(self.parameters['clayer'],(3,3),
                                  activation=self.parameters['activation'],
                                  padding='same',
                                  use_bias=False),
                  BN(),
                  Dropout(self.parameters['dropout']),],
                *[UpSampling2D((2,2)),
                  Deconvolution2D(1,(3,3), activation='sigmoid',padding='same'),],
                Cropping2D(crop),
                Reshape(input_shape),]


# Mixins ################################################################

class ZeroSuppressMixin:
    def _build(self,input_shape):
        super()._build(input_shape)

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["zerosuppress_delay"]):self.parameters["zerosuppress"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))

        zerosuppress_loss = K.mean(self.encoder.output)

        self.net.add_loss(K.in_train_phase(zerosuppress_loss * alpha.variable, 0.0))

        def activation(x, y):
            return zerosuppress_loss

        # def zerosuppres(x, y):
        #     return alpha.variable

        self.metrics.append(activation)
        # self.metrics.append(zerosuppress)
        return


class EarlyStopMixin:
    def _build(self,input_shape):
        super()._build(input_shape)

        # check all hyperparameters and ensure that the earlystop does not activate until all
        # delayed loss epoch kicks in
        max_delay = 0.0
        for key in self.parameters:
            if "delay" in key:
                max_delay = max(max_delay, self.parameters[key])
        self.parameters["earlystop_delay"] = max_delay + 0.1

        self.callbacks.append(
            ChangeEarlyStopping(
                epoch_start=self.parameters["epoch"]*self.parameters["earlystop_delay"],
                verbose=1,))

        self.callbacks.append(
            LinearEarlyStopping(
                self.parameters["epoch"],
                epoch_start = self.parameters["epoch"]*self.parameters["earlystop_delay"],
                value_start = 1.0-self.parameters["earlystop_delay"],
                verbose     = 1,))


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

        self.locality_alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["locality_delay"]):self.parameters["locality"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=self.locality_alpha.update))

        def locality(x, y):
            return self.locality_alpha.variable

        self.metrics.append(locality)
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


# State Auto Encoder ################################################################

class StateAE(EarlyStopMixin, FullConnectedDecoderMixin, FullConnectedEncoderMixin, AE):
    """An AE whose latent layer is GumbelSofmax.
Fully connected layers only, no convolutions.
Note: references to self.parameters[key] are all hyperparameters."""
    def _build(self,input_shape):
        self.encoder_net = self.build_encoder(input_shape)
        self.decoder_net = self.build_decoder(input_shape)

        x = Input(shape=input_shape, name="autoencoder")
        z = Sequential(self.encoder_net)(x)
        y = Sequential(self.decoder_net)(z)

        self.loss = BCE
        self.metrics.append(BCE)
        self.metrics.append(MSE)
        self.encoder     = Model(x, z)
        self.autoencoder = Model(x, y)
        self.net = self.autoencoder
        self.features = Model(x, Sequential([flatten, *_encoder[:-2]])(x))
        self.custom_log_functions['lr'] = lambda: K.get_value(self.net.optimizer.lr)

    def _build_aux(self,input_shape):
        # to be called after the training
        z2 = Input(shape=self.zdim(), name="autodecoder")
        y2 = Sequential(self.decoder_net)(z2)
        w2 = Sequential(self.encoder_net)(y2)
        self.decoder     = Model(z2, y2)
        self.autodecoder = Model(z2, w2)

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


class TransitionAE(ConvolutionalEncoderMixin, StateAE):
    def double_mode(self):
        self.mode(False)
    def single_mode(self):
        self.mode(True)
    def mode(self, single):
        if single:
            if self.built_aux:
                self.encoder     = self.s_encoder
                self.decoder     = self.s_decoder
                self.autoencoder = self.s_autoencoder
                self.autodecoder = self.s_autodecoder
        else:
            self.encoder     = self.d_encoder
            self.autoencoder = self.d_autoencoder
            if self.built_aux:
                self.decoder     = self.d_decoder
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
        self.encoder_net = self.build_encoder(input_shape[1:])
        self.decoder_net = self.build_decoder(input_shape[1:])

        x               = Input(shape=input_shape, name="double_input")
        z, z_pre, z_suc = dapply(x, Sequential(self.encoder_net))
        y, _,     _     = dapply(z, Sequential(self.decoder_net))

        self.d_encoder     = Model(x, z)
        self.d_autoencoder = Model(x, y)

        if "loss" in self.parameters:
            specified = eval(self.parameters["loss"])
            def loss(x,y):
                return K.in_train_phase(specified(x,y), MSE(x,y))

            self.loss = loss
        else:
            self.loss = MSE

        self.net = self.d_autoencoder

        self.double_mode()
        return

    def _build_aux(self,input_shape):
        x = Input(shape=input_shape[1:], name="single_input")
        z = Sequential(self.encoder_net)(x)
        y = Sequential(self.decoder_net)(z)

        z2 = Input(shape=K.int_shape(z)[1:], name="single_input_decoder")
        y2 = Sequential(self.decoder_net)(z2)
        w2 = Sequential(self.encoder_net)(y2)

        self.s_encoder     = Model(x, z)
        self.s_decoder     = Model(z2, y2)
        self.s_autoencoder = Model(x, y)
        self.s_autodecoder = Model(z2, w2)

        z2       = Input(shape=(2,*self.zdim()), name="double_input_decoder")
        y2, _, _ = dapply(z2, Sequential(self.decoder_net))
        w2, _, _ = dapply(y2, Sequential(self.encoder_net))

        self.d_decoder     = Model(z2, y2)
        self.d_autodecoder = Model(z2, w2)

    def dump_actions(self,pre,suc,**kwargs):
        def save(name,data):
            print("Saving to",self.local(name))
            with open(self.local(name), 'wb') as f:
                np.savetxt(f,data,"%d")

        data = np.concatenate([pre,suc],axis=1)
        save("actions.csv", data)
        return


# Transition AE + Action AE double wielding! #################################

class BaseActionMixin:
    def encode_action(self,data,**kwargs):
        return self.action.predict(data,**kwargs)
    def decode_action(self,data,**kwargs):
        return self.apply.predict(data,**kwargs)
    def plot(self,data,path,verbose=False):
        import os.path
        basename, ext = os.path.splitext(path)
        pre_path = basename+"_pre"+ext
        suc_path = basename+"_suc"+ext

        x = data
        z = self.encode(x)
        y = self.decode(z)

        x_pre, x_suc = x[:,0,...], x[:,1,...]
        z_pre, z_suc = z[:,0,...], z[:,1,...]
        y_pre, y_suc = y[:,0,...], y[:,1,...]

        super().plot(x_pre,pre_path,verbose=verbose)
        super().plot(x_suc,suc_path,verbose=verbose)

        action    = self.encode_action(np.concatenate([z_pre,z_suc],axis=1))
        z_suc_aae = self.decode_action([z_pre, action])
        y_suc_aae = self.decode(z_suc_aae)

        z_suc_min = np.minimum(z_suc, z_suc_aae)
        y_suc_min = self.decode(z_suc_min)

        from .util.plot import plot_grid, squarify

        def diff(src,dst):
            return (dst - src + 1)/2
        def _plot(path,columns):
            rows = []
            for seq in zip(*columns):
                rows.extend(seq)
            plot_grid(rows, w=len(columns), path=path, verbose=verbose)

        _z_pre     = squarify(z_pre)
        _z_suc     = squarify(z_suc)
        _z_suc_aae = squarify(z_suc_aae)
        _z_suc_min = squarify(z_suc_min)

        _plot(basename+"_transition"+ext,
              [x_pre, x_suc,
               _z_pre,
               _z_suc,
               _z_suc_aae,
               diff(_z_pre, _z_suc),
               diff(_z_pre, _z_suc_aae),
               diff(_z_suc, _z_suc_aae),
               y_pre,
               y_suc,
               y_suc_aae,
               diff(x_pre,y_pre),
               diff(x_suc,y_suc),
               diff(x_suc,y_suc_aae),])

        return

    def add_metrics(self, x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae, l_pre=None, l_suc=None, l_suc_aae=None, w_suc_aae=None, v_suc_aae=None,):
        # x: inputs
        # l: logits to latent
        # z: latent
        # y: reconstruction

        def mse_x1y1(true,pred):
            return mse(x_pre,y_pre)
        def mse_x2y2(true,pred):
            return mse(x_suc,y_suc)
        def mse_x2y3(true,pred):
            return mse(x_suc,y_suc_aae)
        def mse_y2y3(true,pred):
            return mse(y_suc,y_suc_aae)
        def mse_y2v3(true,pred):
            return mse(y_suc,v_suc_aae)

        def mae_z2z3(true, pred):
            return K.mean(mae(K.round(z_suc), K.round(z_suc_aae)))
        def mae_z2w3(true, pred):
            return K.mean(mae(K.round(z_suc), K.round(w_suc_aae)))

        def mse_l2l3(true, pred):
            return K.mean(mse(l_suc, l_suc_aae))

        def avg_z2(x, y):
            return K.mean(z_suc)
        def avg_z3(x, y):
            return K.mean(z_suc_aae)

        self.metrics.append(mse_x1y1)
        self.metrics.append(mse_x2y2)
        self.metrics.append(mse_x2y3)
        self.metrics.append(mse_y2y3)
        if (v_suc_aae is not None):
            self.metrics.append(mse_y2v3)
        self.metrics.append(mae_z2z3)
        if (w_suc_aae is not None):
            self.metrics.append(mae_z2w3)
        if (l_suc is not None) and (l_suc_aae is not None):
            self.metrics.append(mse_l2l3)

        # self.metrics.append(avg_z2)
        self.metrics.append(avg_z3)

        return
    def build_action_fc_unit(self):
        return Sequential([
            Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
        ])

    def eff_reconstruction_loss(self,x):
        # optional loss, unused
        # _, x_pre, x_suc = dapply(x, lambda x: x)
        # eff_reconstruction_loss = K.mean(bce(x_suc, y_suc_aae))
        # self.net.add_loss(eff_reconstruction_loss)
        return

    def effect_minimization_loss(self):
        # optional loss, unused
        # self.net.add_loss(1*K.mean(K.sum(action_add,axis=-1)))
        # self.net.add_loss(1*K.mean(K.sum(action_del,axis=-1)))

        # depending on how effects are encoded, this is also used
        # self.net.add_loss(1*K.mean(K.sum(action_eff,axis=-1)))
        return

    def _build(self,input_shape):
        super()._build(input_shape)

        x = self.net.input      # keras has a bug, we can't make a new Input here
        _, x_pre, x_suc = dapply(x, lambda x: x)
        z, z_pre, z_suc = dapply(self.d_encoder.output,     lambda x: x)
        y, y_pre, y_suc = dapply(self.d_autoencoder.output, lambda x: x)

        if self.parameters["stop_gradient"]:
            z_pre = wrap(z_pre, K.stop_gradient(z_pre))
            z_suc = wrap(z_suc, K.stop_gradient(z_suc))

        action    = self._action(z_pre,z_suc)
        z_suc_aae = self._apply(z_pre,z_suc,action)
        y_suc_aae = Sequential(self.decoder_net)(z_suc_aae)

        # denoising loop
        v_suc_aae = y_suc_aae
        for i in range(3):
            w_suc_aae = Sequential(self.encoder_net)(v_suc_aae)
            v_suc_aae = Sequential(self.decoder_net)(w_suc_aae)

        self.net = Model(x, dmerge(y_pre, y_suc_aae))

        self.add_metrics(x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae, v_suc_aae=v_suc_aae, w_suc_aae=w_suc_aae)
        return

    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)

        from .util.np_distances import mse, mae

        test_both(["aae","MSE","vanilla"],
                  lambda data: mse(data[:,1,...], self.net.predict(data,          **opts)[:,1,...]))
        test_both(["aae","MSE","gaussian"],
                  lambda data: mse(data[:,1,...], self.net.predict(gaussian(data),**opts)[:,1,...]))
        test_both(["aae","MSE","salt"],
                  lambda data: mse(data[:,1,...], self.net.predict(salt(data),    **opts)[:,1,...]))
        test_both(["aae","MSE","pepper"],
                  lambda data: mse(data[:,1,...], self.net.predict(pepper(data),  **opts)[:,1,...]))

        def true_num_actions(data):
            z     = self.encode(data)
            z2    = z.reshape((-1,2*z.shape[-1]))
            actions = self.encode_action(z2, **opts).round()
            histogram = np.squeeze(actions.sum(axis=0,dtype=int))
            true_num_actions = np.count_nonzero(histogram)
            return true_num_actions

        test_both(["aae","true_num_actions"], true_num_actions)

        def z_mae(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            return mae(z_suc, z_suc_aae)

        test_both(["aae","z_MAE","vanilla"], z_mae)
        test_both(["aae","z_MAE","gaussian"],lambda data: z_mae(gaussian(data)))
        test_both(["aae","z_MAE","salt"],    lambda data: z_mae(salt(data)))
        test_both(["aae","z_MAE","pepper"],  lambda data: z_mae(pepper(data)))

        def z_prob_bitwise(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.prod(np.mean(z_match,axis=0))

        test_both(["aae","z_prob_bitwise","vanilla"], z_prob_bitwise)
        test_both(["aae","z_prob_bitwise","gaussian"],lambda data: z_prob_bitwise(gaussian(data)))
        test_both(["aae","z_prob_bitwise","salt"],    lambda data: z_prob_bitwise(salt(data)))
        test_both(["aae","z_prob_bitwise","pepper"],  lambda data: z_prob_bitwise(pepper(data)))

        def z_allmatch(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.mean(np.prod(z_match,axis=1))

        test_both(["aae","z_allmatch","vanilla"], z_allmatch)
        test_both(["aae","z_allmatch","gaussian"],lambda data: z_allmatch(gaussian(data)))
        test_both(["aae","z_allmatch","salt"],    lambda data: z_allmatch(salt(data)))
        test_both(["aae","z_allmatch","pepper"],  lambda data: z_allmatch(pepper(data)))

        def action_entropy(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)

            A = self.parameters["num_actions"]

            def entropy(j):
                indices = np.nonzero(a[:,0,j])
                z       = z_pre[indices[0]]                     # dimension: [b,N]
                p       = np.mean(z, axis=0)                    # dimension: [N]
                H       = -np.sum(p*np.log(p+1e-20)+(1-p)*np.log(1-p+1e-20)) # dimension: [] (singleton)
                return H

            _H_z_given_a = np.array([entropy(j) for j in range(A)])
            H_z_given_a = np.mean(_H_z_given_a[~np.isnan(_H_z_given_a)])
            return H_z_given_a

        test_both(["H_z_a",], action_entropy)

        return

    def dump_actions(self,pre,suc,**kwargs):
        # data: transition data
        num_actions = self.parameters["num_actions"]
        def to_id(actions):
            return (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

        def save(name,data):
            print("Saving to",self.local(name))
            with open(self.local(name), 'wb') as f:
                np.savetxt(f,data,"%d")

        N=pre.shape[1]
        data = np.concatenate([pre,suc],axis=1)
        actions = self.encode_action(data, **kwargs).round()

        histogram = np.squeeze(actions.sum(axis=0,dtype=int))
        print(histogram)
        true_num_actions = np.count_nonzero(histogram)
        print(true_num_actions)
        all_labels = np.zeros((true_num_actions, actions.shape[1], actions.shape[2]), dtype=int)
        for i, a in enumerate(np.where(histogram > 0)[0]):
            all_labels[i][0][a] = 1

        save("available_actions.csv", np.where(histogram > 0)[0])

        actions_byid = to_id(actions)

        data_byid = np.concatenate((data,actions_byid), axis=1)

        save("actions.csv", data)
        save("actions+ids.csv", data_byid)

        data_aae = np.concatenate([pre,self.decode_action([pre,actions], **kwargs)], axis=1)

        data_aae_byid = np.concatenate((data_aae,actions_byid), axis=1)
        save("actions_aae.csv", data_aae)
        save("actions_aae+ids.csv", data_aae_byid)

        save("actions_both.csv", np.concatenate([data,data_aae], axis=0))
        save("actions_both+ids.csv", np.concatenate([data_byid,data_aae_byid], axis=0))

        # def generate_aae_action(known_transisitons):
        #     states = known_transisitons.reshape(-1, N)
        #     from .util import set_difference
        #     def repeat_over(array, repeats, axis=0):
        #         array = np.expand_dims(array, axis)
        #         array = np.repeat(array, repeats, axis)
        #         return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))
        # 
        #     print("start generating transitions")
        #     # s1,s2,s3,s1,s2,s3,....
        #     repeated_states  = repeat_over(states, len(all_labels), axis=0)
        #     # a1,a1,a1,a2,a2,a2,....
        #     repeated_actions = np.repeat(all_labels, len(states), axis=0)
        # 
        #     y = self.decode_action([repeated_states, repeated_actions], **kwargs).round().astype(np.int8)
        #     y = np.concatenate([repeated_states, y], axis=1)
        # 
        #     print("remove known transitions")
        #     y = set_difference(y, known_transisitons)
        #     print("shuffling")
        #     import numpy.random
        #     numpy.random.shuffle(y)
        #     return y
        # 
        # transitions = generate_aae_action(data)
        # # note: transitions are already shuffled, and also do not contain any examples in data.
        # actions      = self.encode_action(transitions, **kwargs).round()
        # actions_byid = to_id(actions)
        # 
        # # ensure there are enough test examples
        # separation = min(len(data)*10,len(transitions)-len(data))
        # 
        # # fake dataset is used only for the training.
        # fake_transitions  = transitions[:separation]
        # fake_actions_byid = actions_byid[:separation]
        # 
        # # remaining data are used only for the testing.
        # test_transitions  = transitions[separation:]
        # test_actions_byid = actions_byid[separation:]
        # 
        # print(fake_transitions.shape, test_transitions.shape)
        # 
        # save("fake_actions.csv",fake_transitions)
        # save("fake_actions+ids.csv",np.concatenate((fake_transitions,fake_actions_byid), axis=1))
        # 
        # from .util import puzzle_module
        # p = puzzle_module(self.path)
        # print("decoding pre")
        # pre_images = self.decode(test_transitions[:,:N],**kwargs)
        # print("decoding suc")
        # suc_images = self.decode(test_transitions[:,N:],**kwargs)
        # print("validating transitions")
        # valid    = p.validate_transitions([pre_images, suc_images],**kwargs)
        # invalid  = np.logical_not(valid)
        # 
        # valid_transitions  = test_transitions [valid][:len(data)] # reduce the amount of data to reduce runtime
        # valid_actions_byid = test_actions_byid[valid][:len(data)]
        # invalid_transitions  = test_transitions [invalid][:len(data)] # reduce the amount of data to reduce runtime
        # invalid_actions_byid = test_actions_byid[invalid][:len(data)]
        # 
        # save("valid_actions.csv",valid_transitions)
        # save("valid_actions+ids.csv",np.concatenate((valid_transitions,valid_actions_byid), axis=1))
        # save("invalid_actions.csv",invalid_transitions)
        # save("invalid_actions+ids.csv",np.concatenate((invalid_transitions,invalid_actions_byid), axis=1))
        return

class NoSucBaseActionMixin:
    def encode_action(self,data,**kwargs):
        return self.action.predict(data,**kwargs)
    def decode_action(self,data,**kwargs):
        return self.apply.predict(data,**kwargs)
    def plot(self,data,path,verbose=False):
        import os.path
        basename, ext = os.path.splitext(path)
        pre_path = basename+"_pre"+ext
        suc_path = basename+"_suc"+ext

        x = data
        z = self.encode(x)
        y = self.decode(z)

        x_pre, x_suc = x[:,0,...], x[:,1,...]
        z_pre, z_suc = z[:,0,...], z[:,1,...]
        y_pre, y_suc = y[:,0,...], y[:,1,...]

        super().plot(x_pre,pre_path,verbose=verbose)
        super().plot(x_suc,suc_path,verbose=verbose)

        action    = self.encode_action(np.concatenate([z_pre,z_suc],axis=1))
        z_suc_aae = self.decode_action([z_pre, action])
        y_suc_aae = self.decode(z_suc_aae)

        z_suc_min = np.minimum(z_suc, z_suc_aae)
        y_suc_min = self.decode(z_suc_min)

        from .util.plot import plot_grid, squarify

        def diff(src,dst):
            return (dst - src + 1)/2
        def _plot(path,columns):
            rows = []
            for seq in zip(*columns):
                rows.extend(seq)
            plot_grid(rows, w=len(columns), path=path, verbose=verbose)

        _z_pre     = squarify(z_pre)
        _z_suc     = squarify(z_suc)
        _z_suc_aae = squarify(z_suc_aae)
        _z_suc_min = squarify(z_suc_min)

        _plot(basename+"_transition"+ext,
              [x_pre, x_suc,
               _z_pre,
               _z_suc,
               _z_suc_aae,
               diff(_z_pre, _z_suc),
               diff(_z_pre, _z_suc_aae),
               diff(_z_suc, _z_suc_aae),
               y_pre,
               y_suc,
               y_suc_aae,
               diff(x_pre,y_pre),
               diff(x_suc,y_suc),
               diff(x_suc,y_suc_aae),])

        return

    def add_metrics(self, x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae, l_pre=None, l_suc=None, l_suc_aae=None, w_suc_aae=None, v_suc_aae=None,):
        # x: inputs
        # l: logits to latent
        # z: latent
        # y: reconstruction

        def mse_x1y1(true,pred):
            return mse(x_pre,y_pre)
        def mse_x2y2(true,pred):
            return mse(x_suc,y_suc)
        def mse_x2y3(true,pred):
            return mse(x_suc,y_suc_aae)
        def mse_y2y3(true,pred):
            return mse(y_suc,y_suc_aae)
        def mse_y2v3(true,pred):
            return mse(y_suc,v_suc_aae)

        def mae_z2z3(true, pred):
            return K.mean(mae(K.round(z_suc), K.round(z_suc_aae)))
        def mae_z2w3(true, pred):
            return K.mean(mae(K.round(z_suc), K.round(w_suc_aae)))

        def mse_l2l3(true, pred):
            return K.mean(mse(l_suc, l_suc_aae))

        def avg_z2(x, y):
            return K.mean(z_suc)
        def avg_z3(x, y):
            return K.mean(z_suc_aae)

        self.metrics.append(mse_x1y1)
        self.metrics.append(mse_x2y2)
        self.metrics.append(mse_x2y3)
        self.metrics.append(mse_y2y3)
        if (v_suc_aae is not None):
            self.metrics.append(mse_y2v3)
        self.metrics.append(mae_z2z3)
        if (w_suc_aae is not None):
            self.metrics.append(mae_z2w3)
        if (l_suc is not None) and (l_suc_aae is not None):
            self.metrics.append(mse_l2l3)

        # self.metrics.append(avg_z2)
        self.metrics.append(avg_z3)

        return
    def build_action_fc_unit(self):
        return Sequential([
            Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
            BN(),
            Dropout(self.parameters['dropout']),
        ])

    def eff_reconstruction_loss(self,x):
        # optional loss, unused
        # _, x_pre, x_suc = dapply(x, lambda x: x)
        # eff_reconstruction_loss = K.mean(bce(x_suc, y_suc_aae))
        # self.net.add_loss(eff_reconstruction_loss)
        return

    def effect_minimization_loss(self):
        # optional loss, unused
        # self.net.add_loss(1*K.mean(K.sum(action_add,axis=-1)))
        # self.net.add_loss(1*K.mean(K.sum(action_del,axis=-1)))

        # depending on how effects are encoded, this is also used
        # self.net.add_loss(1*K.mean(K.sum(action_eff,axis=-1)))
        return

    def _build(self,input_shape):
        super()._build(input_shape)

        x = self.net.input      # keras has a bug, we can't make a new Input here
        _, x_pre, x_suc = dapply(x, lambda x: x)
        z, z_pre, z_suc = dapply(self.d_encoder.output,     lambda x: x)
        y, y_pre, y_suc = dapply(self.d_autoencoder.output, lambda x: x)

        if self.parameters["stop_gradient"]:
            z_pre = wrap(z_pre, K.stop_gradient(z_pre))
            z_suc = wrap(z_suc, K.stop_gradient(z_suc))

        action    = self._action(z_pre,z_suc)
        z_suc_aae = self._apply(z_pre,z_suc,action)
        y_suc_aae = Sequential(self.decoder_net)(wrap(z_suc, z_suc + 0.0 * z_suc_aae))

        # denoising loop
        v_suc_aae = y_suc_aae
        for i in range(3):
            w_suc_aae = Sequential(self.encoder_net)(v_suc_aae)
            v_suc_aae = Sequential(self.decoder_net)(w_suc_aae)

        # do not optimize the successor image
        self.net = Model(x, dmerge(y_pre, y_suc_aae))

        self.add_metrics(x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae, v_suc_aae=v_suc_aae, w_suc_aae=w_suc_aae)
        return

    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)

        from .util.np_distances import mse, mae

        test_both(["aae","MSE","vanilla"],
                  lambda data: mse(data[:,1,...], self.net.predict(data,          **opts)[:,1,...]))
        test_both(["aae","MSE","gaussian"],
                  lambda data: mse(data[:,1,...], self.net.predict(gaussian(data),**opts)[:,1,...]))
        test_both(["aae","MSE","salt"],
                  lambda data: mse(data[:,1,...], self.net.predict(salt(data),    **opts)[:,1,...]))
        test_both(["aae","MSE","pepper"],
                  lambda data: mse(data[:,1,...], self.net.predict(pepper(data),  **opts)[:,1,...]))

        def true_num_actions(data):
            z     = self.encode(data)
            z2    = z.reshape((-1,2*z.shape[-1]))
            actions = self.encode_action(z2, **opts).round()
            histogram = np.squeeze(actions.sum(axis=0,dtype=int))
            true_num_actions = np.count_nonzero(histogram)
            return true_num_actions

        test_both(["aae","true_num_actions"], true_num_actions)

        def z_mae(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            return mae(z_suc, z_suc_aae)

        test_both(["aae","z_MAE","vanilla"], z_mae)
        test_both(["aae","z_MAE","gaussian"],lambda data: z_mae(gaussian(data)))
        test_both(["aae","z_MAE","salt"],    lambda data: z_mae(salt(data)))
        test_both(["aae","z_MAE","pepper"],  lambda data: z_mae(pepper(data)))

        def z_prob_bitwise(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.prod(np.mean(z_match,axis=0))

        test_both(["aae","z_prob_bitwise","vanilla"], z_prob_bitwise)
        test_both(["aae","z_prob_bitwise","gaussian"],lambda data: z_prob_bitwise(gaussian(data)))
        test_both(["aae","z_prob_bitwise","salt"],    lambda data: z_prob_bitwise(salt(data)))
        test_both(["aae","z_prob_bitwise","pepper"],  lambda data: z_prob_bitwise(pepper(data)))

        def z_allmatch(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)
            z_suc_aae = self.decode_action([z_pre,a], **opts)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.mean(np.prod(z_match,axis=1))

        test_both(["aae","z_allmatch","vanilla"], z_allmatch)
        test_both(["aae","z_allmatch","gaussian"],lambda data: z_allmatch(gaussian(data)))
        test_both(["aae","z_allmatch","salt"],    lambda data: z_allmatch(salt(data)))
        test_both(["aae","z_allmatch","pepper"],  lambda data: z_allmatch(pepper(data)))

        def action_entropy(data):
            z     = self.encode(data)
            z_pre = z[:,0,...]
            z_suc = z[:,1,...]
            z2    = z.reshape((-1,2*z.shape[-1]))
            a     = self.encode_action(z2,**opts)

            A = self.parameters["num_actions"]

            def entropy(j):
                indices = np.nonzero(a[:,0,j])
                z       = z_pre[indices[0]]                     # dimension: [b,N]
                p       = np.mean(z, axis=0)                    # dimension: [N]
                H       = -np.sum(p*np.log(p+1e-20)+(1-p)*np.log(1-p+1e-20)) # dimension: [] (singleton)
                return H

            _H_z_given_a = np.array([entropy(j) for j in range(A)])
            H_z_given_a = np.mean(_H_z_given_a[~np.isnan(_H_z_given_a)])
            return H_z_given_a

        test_both(["H_z_a",], action_entropy)

        return

    def dump_actions(self,pre,suc,**kwargs):
        # data: transition data
        num_actions = self.parameters["num_actions"]
        def to_id(actions):
            return (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

        def save(name,data):
            print("Saving to",self.local(name))
            with open(self.local(name), 'wb') as f:
                np.savetxt(f,data,"%d")

        N=pre.shape[1]
        data = np.concatenate([pre,suc],axis=1)
        actions = self.encode_action(data, **kwargs).round()

        histogram = np.squeeze(actions.sum(axis=0,dtype=int))
        print(histogram)
        true_num_actions = np.count_nonzero(histogram)
        print(true_num_actions)
        all_labels = np.zeros((true_num_actions, actions.shape[1], actions.shape[2]), dtype=int)
        for i, a in enumerate(np.where(histogram > 0)[0]):
            all_labels[i][0][a] = 1

        save("available_actions.csv", np.where(histogram > 0)[0])

        actions_byid = to_id(actions)

        data_byid = np.concatenate((data,actions_byid), axis=1)

        save("actions.csv", data)
        save("actions+ids.csv", data_byid)

        data_aae = np.concatenate([pre,self.decode_action([pre,actions], **kwargs)], axis=1)

        data_aae_byid = np.concatenate((data_aae,actions_byid), axis=1)
        save("actions_aae.csv", data_aae)
        save("actions_aae+ids.csv", data_aae_byid)

        save("actions_both.csv", np.concatenate([data,data_aae], axis=0))
        save("actions_both+ids.csv", np.concatenate([data_byid,data_aae_byid], axis=0))

        # def generate_aae_action(known_transisitons):
        #     states = known_transisitons.reshape(-1, N)
        #     from .util import set_difference
        #     def repeat_over(array, repeats, axis=0):
        #         array = np.expand_dims(array, axis)
        #         array = np.repeat(array, repeats, axis)
        #         return np.reshape(array,(*array.shape[:axis],-1,*array.shape[axis+2:]))
        # 
        #     print("start generating transitions")
        #     # s1,s2,s3,s1,s2,s3,....
        #     repeated_states  = repeat_over(states, len(all_labels), axis=0)
        #     # a1,a1,a1,a2,a2,a2,....
        #     repeated_actions = np.repeat(all_labels, len(states), axis=0)
        # 
        #     y = self.decode_action([repeated_states, repeated_actions], **kwargs).round().astype(np.int8)
        #     y = np.concatenate([repeated_states, y], axis=1)
        # 
        #     print("remove known transitions")
        #     y = set_difference(y, known_transisitons)
        #     print("shuffling")
        #     import numpy.random
        #     numpy.random.shuffle(y)
        #     return y
        # 
        # transitions = generate_aae_action(data)
        # # note: transitions are already shuffled, and also do not contain any examples in data.
        # actions      = self.encode_action(transitions, **kwargs).round()
        # actions_byid = to_id(actions)
        # 
        # # ensure there are enough test examples
        # separation = min(len(data)*10,len(transitions)-len(data))
        # 
        # # fake dataset is used only for the training.
        # fake_transitions  = transitions[:separation]
        # fake_actions_byid = actions_byid[:separation]
        # 
        # # remaining data are used only for the testing.
        # test_transitions  = transitions[separation:]
        # test_actions_byid = actions_byid[separation:]
        # 
        # print(fake_transitions.shape, test_transitions.shape)
        # 
        # save("fake_actions.csv",fake_transitions)
        # save("fake_actions+ids.csv",np.concatenate((fake_transitions,fake_actions_byid), axis=1))
        # 
        # from .util import puzzle_module
        # p = puzzle_module(self.path)
        # print("decoding pre")
        # pre_images = self.decode(test_transitions[:,:N],**kwargs)
        # print("decoding suc")
        # suc_images = self.decode(test_transitions[:,N:],**kwargs)
        # print("validating transitions")
        # valid    = p.validate_transitions([pre_images, suc_images],**kwargs)
        # invalid  = np.logical_not(valid)
        # 
        # valid_transitions  = test_transitions [valid][:len(data)] # reduce the amount of data to reduce runtime
        # valid_actions_byid = test_actions_byid[valid][:len(data)]
        # invalid_transitions  = test_transitions [invalid][:len(data)] # reduce the amount of data to reduce runtime
        # invalid_actions_byid = test_actions_byid[invalid][:len(data)]
        # 
        # save("valid_actions.csv",valid_transitions)
        # save("valid_actions+ids.csv",np.concatenate((valid_transitions,valid_actions_byid), axis=1))
        # save("invalid_actions.csv",invalid_transitions)
        # save("invalid_actions+ids.csv",np.concatenate((invalid_transitions,invalid_actions_byid), axis=1))
        return


# action mapping variations

class DetActionMixin:
    "Deterministic mapping from a state pair to an action"
    def build_action_encoder(self):
        return [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                        Dense(self.parameters['num_actions']),
                        self.build_gs(N=1,
                                      M=self.parameters['num_actions'],
                                      offset=self.parameters["aae_delay"],),
            ]),
        ]
    def _action(self,z_pre,z_suc):
        self.action_encoder_net = self.build_action_encoder()

        N = self.parameters['N']
        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        suc2 = wrap(transition, transition[:,N:])
        self.action = Model(transition, ConditionalSequential(self.action_encoder_net, pre2, axis=1)(suc2))

        return ConditionalSequential(self.action_encoder_net, z_pre, axis=1)(z_suc)


# effect mapping variations

class DirectLossMixin:
    "Additional loss for the latent space successor prediction."
    def _build(self,input_shape):
        super()._build(input_shape)

        self.direct_alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["direct_delay"]):self.parameters["direct"],
        })
        self.callbacks.append(LambdaCallback(on_epoch_end=self.direct_alpha.update))

        return
    def apply_direct_loss(self,true,pred):
        dummy = Lambda(lambda x: x)

        loss = K.mean(mae(true, pred))
        def direct(x, y):
            return loss

        self.metrics.append(direct)

        # direct loss should be treated as the real loss
        dummy.add_loss(K.in_train_phase(loss * self.direct_alpha.variable, loss))
        return dummy(pred)

class ActionDumpMixin:
    def dump_actions(self,pre,suc,**kwargs):
        # data: transition data
        num_actions = self.parameters["num_actions"]
        def to_id(actions):
            return (actions * np.arange(num_actions)).sum(axis=-1,dtype=int)

        def save(name,data):
            print("Saving to",self.local(name))
            with open(self.local(name), 'wb') as f:
                np.savetxt(f,data,"%d")

        N=pre.shape[1]
        data = np.concatenate([pre,suc],axis=1)
        actions = self.encode_action(data, **kwargs).round()

        histogram = np.squeeze(actions.sum(axis=0,dtype=int))
        print(histogram)
        true_num_actions = np.count_nonzero(histogram)
        print(true_num_actions)
        all_labels = np.zeros((true_num_actions, actions.shape[1], actions.shape[2]), dtype=int)
        for i, a in enumerate(np.where(histogram > 0)[0]):
            all_labels[i][0][a] = 1

        save("available_actions.csv", np.where(histogram > 0)[0])

        actions_byid = to_id(actions)

        data_byid = np.concatenate((data,actions_byid), axis=1)

        save("actions.csv", data)
        save("actions+ids.csv", data_byid)

        data_aae = np.concatenate([pre,self.decode_action([pre,actions], **kwargs)], axis=1)

        data_aae_byid = np.concatenate((data_aae,actions_byid), axis=1)
        save("actions_aae.csv", data_aae)
        save("actions_aae+ids.csv", data_aae_byid)

        save("actions_both.csv", np.concatenate([data,data_aae], axis=0))
        save("actions_both+ids.csv", np.concatenate([data_byid,data_aae_byid], axis=0))

        all_actions_byid = to_id(all_labels)

        def extract_effect_from_transitions(transitions):
            pre = transitions[:,:N]
            suc = transitions[:,N:]
            data_diff = suc - pre
            data_add  = np.maximum(0, data_diff)
            data_del  = -np.minimum(0, data_diff)

            add_effect = np.zeros((true_num_actions, N))
            del_effect = np.zeros((true_num_actions, N))

            for i, a in enumerate(np.where(histogram > 0)[0]):
                indices = np.where(actions_byid == a)[0]
                add_effect[i] = np.amax(data_add[indices], axis=0)
                del_effect[i] = np.amax(data_del[indices], axis=0)

            return add_effect, del_effect, data_diff

        # effects obtained from the latent vectors
        add_effect2, del_effect2, diff2 = extract_effect_from_transitions(data)

        save("action_add2.csv",add_effect2)
        save("action_del2.csv",del_effect2)
        save("action_add2+ids.csv",np.concatenate((add_effect2,all_actions_byid), axis=1))
        save("action_del2+ids.csv",np.concatenate((del_effect2,all_actions_byid), axis=1))
        save("diff2+ids.csv",np.concatenate((diff2,actions_byid), axis=1))

        data_aae = np.concatenate([pre,self.decode_action([pre,actions], **kwargs)], axis=1)

        # effects obtained from the latent vectors, but the successor uses the ones coming from the AAE
        add_effect3, del_effect3, diff3 = extract_effect_from_transitions(data_aae)

        save("action_add3.csv",add_effect3)
        save("action_del3.csv",del_effect3)
        save("action_add3+ids.csv",np.concatenate((add_effect3,all_actions_byid), axis=1))
        save("action_del3+ids.csv",np.concatenate((del_effect3,all_actions_byid), axis=1))
        save("diff3+ids.csv",np.concatenate((diff3,actions_byid), axis=1))

        return

class ConditionalEffectMixin(BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends on both the current state and the action labels -- Same as AAE in AAAI18."
    def _apply(self,z_pre,z_suc,action):

        self.action_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())),
                rounded_sigmoid(),
                Reshape(self.zdim()),
            ])
        ]

        z_suc_aae = ConditionalSequential(self.action_decoder_net, z_pre, axis=1)(flatten(action))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        self.apply  = Model([pre2,act2], ConditionalSequential(self.action_decoder_net, pre2, axis=1)(flatten(act2)))

        return z_suc_aae

class BoolMinMaxEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())*3),
                Reshape((*self.zdim(),3)),
                rounded_softmax(),
                Reshape((*self.zdim(),3)),
            ])
        ]

        z_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        z_add     = wrap(z_eff, z_eff[...,0])
        z_del     = wrap(z_eff, z_eff[...,1])
        z_suc_aae = wrap(z_pre, K.minimum(1-z_del, K.maximum(z_add, z_pre)))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, K.minimum(1-del2, K.maximum(add2, pre2))))
        return z_suc_aae

class BoolSmoothMinMaxEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary smooth min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())*3),
                Reshape((*self.zdim(),3)),
                rounded_softmax(),
                Reshape((*self.zdim(),3)),
            ])
        ]

        z_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        z_add     = wrap(z_eff, z_eff[...,0])
        z_del     = wrap(z_eff, z_eff[...,1])
        z_suc_aae = wrap(z_pre, smooth_min(1-z_del, smooth_max(z_add, z_pre)))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, smooth_min(1-del2, smooth_max(add2, pre2))))
        return z_suc_aae

class BoolAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary smooth min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())*3),
                Reshape((*self.zdim(),3)),
                rounded_softmax(),
                Reshape((*self.zdim(),3)),
            ])
        ]

        z_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        z_add     = wrap(z_eff, z_eff[...,0])
        z_del     = wrap(z_eff, z_eff[...,1])
        z_suc_aae = wrap(z_pre, rounded_sigmoid()(z_pre - 0.5 + z_add - z_del))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, rounded_sigmoid()(pre2 - 0.5 + add2 - del2)))
        return z_suc_aae

class NormalizedLogitAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique with batchnorm."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())),
                BN(),
            ])
        ]
        self.scaling = BN()

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = rounded_sigmoid()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = rounded_sigmoid()(lsuc2)

        self.apply  = Model([pre2,act2], suc2)
        return z_suc_aae

class LogitAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique, but without batchnorm (s_t shifted from [0,1] to [-1/2,1/2].)"
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())),
            ])
        ]
        self.scaling = Lambda(lambda x: 2*x - 1) # same scale as the final batchnorm

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = rounded_sigmoid()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = rounded_sigmoid()(lsuc2)

        self.apply  = Model([pre2,act2], suc2)
        return z_suc_aae

class LogitAddEffect2Mixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique. Uses batchnorm in effect, but not in s_t (shifted from [0,1] to [-1/2,1/2].)"
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())),
                BN(),
            ])
        ]
        self.scaling = Lambda(lambda x: 2*x - 1)

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = rounded_sigmoid()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = rounded_sigmoid()(lsuc2)

        self.apply  = Model([pre2,act2], suc2)
        return z_suc_aae

class NoSucNormalizedLogitAddEffectMixin(ActionDumpMixin,NoSucBaseActionMixin,DirectLossMixin,HammingLoggerMixin):
    """Same as NormalizedLogitAddEffectMixin, but the action prediction takes only the current state.
Code-wise, there is only the inheritance difference.
The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique with batchnorm."""
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"])],
            Sequential([
                Dense(np.prod(self.zdim())),
                BN(),
            ])
        ]
        self.scaling = BN()

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = rounded_sigmoid()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)

        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters['num_actions'],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = rounded_sigmoid()(lsuc2)

        self.apply  = Model([pre2,act2], suc2)
        return z_suc_aae



# Zero-sup SAE ###############################################################

class ConvolutionalStateAE(ConvolutionalEncoderMixin, ConcreteLatentMixin, StateAE):
    pass
class Convolutional2StateAE(ConvolutionalDecoderMixin, ConvolutionalEncoderMixin, ConcreteLatentMixin, StateAE):
    pass

class ZeroSuppressStateAE(ZeroSuppressMixin, ConcreteLatentMixin, StateAE):
    pass
class ZeroSuppressConvolutionalStateAE(ZeroSuppressMixin, ConvolutionalEncoderMixin, ConcreteLatentMixin, StateAE):
    pass
class ZeroSuppressConvolutional2StateAE(ZeroSuppressMixin, ConvolutionalDecoderMixin, ConvolutionalEncoderMixin, ConcreteLatentMixin, StateAE):
    pass
# Transition SAE ################################################################

class VanillaTransitionAE(              ZeroSuppressMixin, ConcreteLatentMixin, TransitionAE):
    pass

# earlier attempts to "sparcify" the transisions. No longer used
class HammingTransitionAE(HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, TransitionAE):
    pass
class CosineTransitionAE (CosineMixin,  ZeroSuppressMixin, ConcreteLatentMixin, TransitionAE):
    pass
class PoissonTransitionAE(PoissonMixin, ZeroSuppressMixin, ConcreteLatentMixin, TransitionAE):
    pass


# IJCAI2020 papers
class ConcreteDetConditionalEffectTransitionAE              (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, ConditionalEffectMixin, TransitionAE):
    pass
class ConcreteDetBoolMinMaxEffectTransitionAE               (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolMinMaxEffectMixin, TransitionAE):
    pass
class ConcreteDetBoolSmoothMinMaxEffectTransitionAE         (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolSmoothMinMaxEffectMixin, TransitionAE):
    pass
class ConcreteDetBoolAddEffectTransitionAE                  (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolAddEffectMixin, TransitionAE):
    pass
class ConcreteDetLogitAddEffectTransitionAE                 (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, LogitAddEffectMixin, TransitionAE):
    pass
class ConcreteDetLogitAddEffect2TransitionAE                (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, LogitAddEffect2Mixin, TransitionAE):
    pass
class ConcreteDetNormalizedLogitAddEffectTransitionAE       (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, NormalizedLogitAddEffectMixin, TransitionAE):
    pass
class ConcreteDetNoSucNormalizedLogitAddEffectTransitionAE  (HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, NoSucNormalizedLogitAddEffectMixin, TransitionAE):
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

    def _report(self,test_both,**opts):
        from .util.np_distances import bce
        test_both(["BCE"], lambda data: bce(data, self.autoencode(data,**opts)))
        return
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
              val_data=None,
              val_data_to=None,
              **kwargs):
        super().train(train_data,
                      batch_size=batch_size,
                      train_data_to=train_data_to,
                      val_data=val_data,
                      val_data_to=val_data_to,
                      save=False,
                      **kwargs)

        s = self.net.predict(val_data[val_data_to == 1],batch_size=batch_size)
        if np.count_nonzero(val_data_to == 1) > 0:
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
                    Dense(self.parameters['aae_width'], activation=self.parameters['aae_activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),])
                for i in range(self.parameters['aae_depth'])
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
                    Dense(self.parameters['aae_width'], activation=self.parameters['aae_activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),])
                for i in range(self.parameters['aae_depth'])
            ],
            Sequential([
                Dense(data_dim),
                rounded_sigmoid(),
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

class CubeActionAE(ActionAE):
    """AAE with cube-like structure, developped for a compariason purpose."""
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters['aae_width'], activation=self.parameters['aae_activation'], use_bias=False),
                    BN(),
                    Dropout(self.parameters['dropout']),])
                for i in range(self.parameters['aae_depth'])
            ],
            Sequential([
                Dense(data_dim,use_bias=False),
                BN(),
            ]),
        ]

    def _build(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters['M'], self.parameters['N']

        x = Input(shape=input_shape)

        pre = wrap(x,x[:,:dim],name="pre")
        suc = wrap(x,x[:,dim:],name="suc")

        _encoder = self.build_encoder([dim])
        action = ConditionalSequential(_encoder, pre, axis=1)(suc)

        _decoder = self.build_decoder([dim])
        l_eff = Sequential(_decoder)(flatten(action))

        scaling = BN()
        l_pre = scaling(pre)

        l_suc = add([l_eff,l_pre])
        suc_reconstruction = rounded_sigmoid()(l_suc)

        y = Concatenate(axis=1)([pre,suc_reconstruction])

        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        l_pre2 = scaling(pre2)
        l_eff2 = Sequential(_decoder)(flatten(action2))
        l_suc2 = add([l_eff2,l_pre2])
        suc_reconstruction2 = rounded_sigmoid()(l_suc2)
        y2 = Concatenate(axis=1)([pre2,suc_reconstruction2])

        self.metrics.append(MAE)
        self.loss = BCE
        self.encoder     = Model(x, [pre,action])
        self.decoder     = Model([pre2,action2], y2)

        self.net = Model(x, y)
        self.autoencoder = self.net


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
              val_data=None,
              val_data_to=None,
              *args,**kwargs):

        self.build(train_data.shape[1:])

        num   = len(val_data_to)
        num_p = np.count_nonzero(val_data_to)
        num_n = num-num_p
        assert num_n > num_p
        print("positive : negative = ",num_p,":",num_n,"negative ratio",num_n/num_p)

        ind_p = np.where(val_data_to == 1)[0]
        ind_n = np.where(val_data_to == 0)[0]

        from numpy.random import shuffle
        shuffle(ind_n)

        per_bag = num_n // len(self.discriminators)
        for i, d in enumerate(self.discriminators):
            print("training",i+1,"/",len(self.discriminators),"th discriminator")
            ind_n_per_bag = ind_n[per_bag*i:per_bag*(i+1)]
            ind = np.concatenate((ind_p,ind_n_per_bag))
            d.train(train_data[ind],
                    train_data_to=train_data_to[ind],
                    val_data=val_data,
                    val_data_to=val_data_to,
                    *args,**kwargs)

    def discriminate(self,data,**kwargs):
        return self.net.predict(data,**kwargs)


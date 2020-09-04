#!/usr/bin/env python3

"""
Model classes for latplan.
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
from .util.plot      import plot_grid, squarify
from .util           import ensure_list, NpEncoder, curry
from .util.stacktrace import print_object
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


def _plot(path,columns):
    rows = []
    for seq in zip(*columns):
        rows.extend(seq)
    plot_grid(rows, w=len(columns), path=path, verbose=True)
    return

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
                          keras.callbacks.TensorBoard(log_dir=self.local("logs/{}-{}".format(path,datetime.datetime.now().isoformat())), write_graph=False)]

    def build(self,*args,**kwargs):
        """An interface for building a network. Input-shape: list of dimensions.
Users should not overload this method; Define _build() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built:
            print("Avoided building {} twice.".format(self))
            return
        print("Building the network")
        self._build(*args,**kwargs)
        self.built = True
        print("Network built")
        return self

    def _build(self,*args,**kwargs):
        """An interface for building a network.
This function is called by build() only when the network is not build yet.
Users may define a method for each subclass for adding a new build-time feature.
Each method should call the _build() method of the superclass in turn.
Users are not expected to call this method directly. Call build() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        return self._build_primary(*args,**kwargs)

    def _build_primary(self,*args,**kwargs):
        pass

    def build_aux(self,*args,**kwargs):
        """An interface for building an additional network not required for training.
To be used after the training.
Input-shape: list of dimensions.
Users should not overload this method; Define _build() for each subclass instead.
This function calls _build bottom-up from the least specialized class.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        if self.built_aux:
            print("Avoided building {} twice.".format(self))
            return
        print("Building the auxiliary network")
        self._build_aux(*args,**kwargs)
        self.built_aux = True
        print("Auxiliary network built")
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
        return self._build_aux_primary(*args,**kwargs)

    def _build_aux_primary(self,*args,**kwargs):
        pass

    def compile(self,*args,**kwargs):
        """An interface for compiling a network."""
        if self.compiled:
            print("Avoided compiling {} twice.".format(self))
            return
        print("Compiling the network")
        self._compile(*args,**kwargs)
        self.compiled = True
        print("Network compiled")
        return self

    def _compile(self,optimizers):
        """An interface for compileing a network."""
        # default method.
        print(f"there are {len(self.nets)} networks.")
        print(f"there are {len(optimizers)} optimizers.")
        print(f"there are {len(self.losses)} losses.")
        assert len(self.nets) == len(optimizers)
        assert len(self.nets) == len(self.losses)
        for net, o, loss in zip(self.nets, optimizers, self.losses):
            print(f"compiling {net} with {o}, {loss}.")
            net.compile(optimizer=o, loss=loss, metrics=self.metrics)
        return

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
        print("Saving the network to {}".format(self.local("")))
        self._save()
        print("Network saved")
        return self

    def _save(self):
        """An interface for saving a network.
Users may define a method for each subclass for adding a new save-time feature.
Each method should call the _save() method of the superclass in turn.
Users are not expected to call this method directly. Call save() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        for i, net in enumerate(self.nets):
            net.save_weights(self.local("net{}.h5".format(i)))

        with open(self.local("aux.json"), "w") as f:
            json.dump({"parameters":self.parameters,
                       "class"     :self.__class__.__name__,
                       "input_shape":self.net.input_shape[1:]}, f , skipkeys=True, cls=NpEncoder, indent=2)

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
        if self.loaded:
            print("Avoided loading {} twice.".format(self))
            return

        if allow_failure:
            try:
                print("Loading the network from {} (with failure allowed)".format(self.local("")))
                self._load()
                self.loaded = True
                print("Network loaded")
            except Exception as e:
                print("Exception {} during load(), ignored.".format(e))
        else:
            print("Loading the network from {} (with failure not allowed)".format(self.local("")))
            self._load()
            self.loaded = True
            print("Network loaded")
        return self

    def _load(self):
        """An interface for loading a network.
Users may define a method for each subclass for adding a new load-time feature.
Each method should call the _load() method of the superclass in turn.
Users are not expected to call this method directly. Call load() instead.
Poor python coders cannot enjoy the cleanness of CLOS :before, :after, :around methods."""
        with open(self.local("aux.json"), "r") as f:
            data = json.load(f)
            self.parameters = data["parameters"]
            self.build(tuple(data["input_shape"]))
            self.build_aux(tuple(data["input_shape"]))
        for i, net in enumerate(self.nets):
            net.load_weights(self.local("net{}.h5".format(i)))

    def initialize_bar(self):
        import progressbar
        widgets = [
            progressbar.Timer(format="%(elapsed)s"),
            " ", progressbar.Counter(), " | ",
            # progressbar.Bar(),
            progressbar.AbsoluteETA(format="%(eta)s"), " ",
            DynamicMessage("status")
        ]
        self.bar = progressbar.ProgressBar(max_value=self.max_epoch, widgets=widgets)

    def bar_update(self, epoch, logs):
        "Used for updating the progress bar."

        if not hasattr(self,"bar"):
            self.initialize_bar()
            from colors import color
            from functools import partial
            self.style = partial(color, fg="black", bg="white")

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
            self.bar.update(epoch+1, status = self.style("[v] "+"  ".join(["{} {:8.3g}".format(k,v) for k,v in sorted(vlogs.items())])) + "\n")
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
              epoch=200,batch_size=1000,optimizer="adam",lr=0.0001,val_data=None,save=True,
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
        input_shape = train_data.shape[1:]
        self.build(input_shape)

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
            "batch_size": batch_size,
            "epochs": epoch,
            "steps": None,
            "samples": len(train_data[0]),
            "verbose": 0,
            "do_validation": False,
            "metrics": [],
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
        try:
            self.build_aux(input_shape)
        except Exception as e:
            print("building the auxilialy network failed.")
            from .util.stacktrace import format
            format(False)
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
        opts = {"verbose":0,"batch_size":batch_size}

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

        with open(self.local("performance.json"), "w") as f:
            json.dump(performance, f, cls=NpEncoder, indent=2)

        with open(self.local("parameter_count.json"), "w") as f:
            json.dump(count_params(self.autoencoder), f, cls=NpEncoder, indent=2)

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

        def fn(N               = self.parameters["N"],
               M               = self.parameters["M"],
               max_temperature = self.parameters["max_temperature"],
               min_temperature = self.parameters["min_temperature"],
               full_epoch      = self.parameters["full_epoch"],
               train_noise     = self.parameters["train_noise"],
               train_hard      = self.parameters["train_hard"],
               test_noise      = self.parameters["test_noise"],
               test_hard       = self.parameters["test_hard"],
               beta            = self.parameters["beta"],
               offset          = 0):
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,full_epoch,
                offset      = offset,
                train_noise = train_noise,
                train_hard  = train_hard,
                test_noise  = test_noise,
                test_hard   = test_hard,
                beta        = beta)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            # self.custom_log_functions["tau"] = lambda: K.get_value(gs.variable)
            return gs

        return fn(**kwargs)

    def build_bc(self,
                 **kwargs):
        # python methods cannot use self in the
        # default values, because python sucks

        def fn(max_temperature = self.parameters["max_temperature"],
               min_temperature = self.parameters["min_temperature"],
               full_epoch      = self.parameters["full_epoch"],
               train_noise     = self.parameters["train_noise"],
               train_hard      = self.parameters["train_hard"],
               test_noise      = self.parameters["test_noise"],
               test_hard       = self.parameters["test_hard"],
               beta            = self.parameters["beta"],
               offset          = 0):
            bc = BinaryConcrete(
                min_temperature,max_temperature,full_epoch,
                offset      = offset,
                train_noise = train_noise,
                train_hard  = train_hard,
                test_noise  = test_noise,
                test_hard   = test_hard,
                beta        = beta)
            self.callbacks.append(LambdaCallback(on_epoch_end=bc.update))
            # self.custom_log_functions["tau"] = lambda: K.get_value(gs.variable)
            return bc

        return fn(**kwargs)

# Latent Activations ################################################################

class ConcreteLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters["M"] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters["N"],)
    def zindim(self):
        return (self.parameters["N"],)
    def activation(self):
        return self.build_bc()

class QuantizedLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters["M"] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters["N"],)
    def zindim(self):
        return (self.parameters["N"],)
    def activation(self):
        return heavyside()

class SigmoidLatentMixin:
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # this is not necessary for shape consistency but it helps pruning some hyperparameters
        if "parameters" in kwargs: # otherwise the object is instantiated without paramteters for loading the value later
            if self.parameters["M"] != 2:
                raise InvalidHyperparameterError()
    def zdim(self):
        return (self.parameters["N"],)
    def zindim(self):
        return (self.parameters["N"],)
    def activation(self):
        return rounded_sigmoid()

class GumbelSoftmaxLatentMixin:
    def zdim(self):
        return (self.parameters["N"]*self.parameters["M"],)
    def zindim(self):
        return (self.parameters["N"]*self.parameters["M"],)
    def activation(self):
        return Sequential([
            self.build_gs(),
            flatten,
        ])

class SoftmaxLatentMixin:
    def zdim(self):
        return (self.parameters["N"]*self.parameters["M"],)
    def zindim(self):
        return (self.parameters["N"]*self.parameters["M"],)
    def activation(self):
        return Sequential([
            Reshape((self.parameters["N"],self.parameters["M"],)),
            rounded_softmax(),
            flatten,
        ])

# Encoder / Decoder ################################################################

class FullConnectedEncoderMixin:
    def build_encoder(self,input_shape):
        return [flatten,
                GaussianNoise(self.parameters["noise"]),
                BN(),
                Dense(self.parameters["layer"], activation="relu", use_bias=False),
                BN(),
                Dropout(self.parameters["dropout"]),
                Dense(self.parameters["layer"], activation="relu", use_bias=False),
                BN(),
                Dropout(self.parameters["dropout"]),
                Dense(self.parameters["layer"], activation="relu", use_bias=False),
                BN(),
                Dropout(self.parameters["dropout"]),
                Dense(np.prod(self.zindim())),
                self.activation(),
        ]

class FullConnectedDecoderMixin:
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            flatten,
            *([Dropout(self.parameters["dropout"])] if self.parameters["dropout_z"] else []),
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(data_dim, activation="sigmoid"),
            Reshape(input_shape),]

class ConvolutionalEncoderMixin:
    """A mixin that uses convolutions + fc in the encoder."""
    def build_encoder(self,input_shape):
        if len(input_shape) == 2:
            reshape = Reshape((*input_shape,1)) # monochrome image
        elif len(input_shape) == 3:
            reshape = lambda x: x
        else:
            raise Exception(f"ConvolutionalEncoderMixin: unsupported shape {input_shape}")

        return [reshape,
                GaussianNoise(self.parameters["noise"]),
                BN(),
                *[Convolution2D(self.parameters["clayer"],(3,3),
                                activation="relu",padding="same", use_bias=False),
                  Dropout(self.parameters["dropout"]),
                  BN(),
                  MaxPooling2D((2,2)),],
                *[Convolution2D(self.parameters["clayer"],(3,3),
                                activation="relu",padding="same", use_bias=False),
                  Dropout(self.parameters["dropout"]),
                  BN(),
                  MaxPooling2D((2,2)),],
                flatten,
                Sequential([
                    Dense(self.parameters["layer"], activation="relu", use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),
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

        return [*([Dropout(self.parameters["dropout"])] if self.parameters["dropout_z"] else []),
                *[Dense(self.parameters["layer"],
                        activation="relu",
                        use_bias=False),
                  BN(),
                  Dropout(self.parameters["dropout"]),],
                *[Dense(np.prod(last_convolution) * self.parameters["clayer"],
                        activation="relu",
                        use_bias=False),
                  BN(),
                  Dropout(self.parameters["dropout"]),],
                Reshape((*last_convolution, self.parameters["clayer"])),
                *[UpSampling2D((2,2)),
                  Deconvolution2D(self.parameters["clayer"],(3,3),
                                  activation="relu",
                                  padding="same",
                                  use_bias=False),
                  BN(),
                  Dropout(self.parameters["dropout"]),],
                *[UpSampling2D((2,2)),
                  Deconvolution2D(1,(3,3), activation="sigmoid",padding="same"),],
                Cropping2D(crop),
                Reshape(input_shape),]


class FullyConvolutionalAEMixin:
    """A mixin that uses only convolutional layers in the encoder/decoder."""

    def output_shape(self,layers,input_shape):
        from functools import reduce
        def c(input_shape,layer):
            print(input_shape)
            return layer.compute_output_shape(input_shape)
        return reduce(c,layers,input_shape)

    def encoder_block(self,i):
        """Extend this method for Residual Nets"""
        k = self.parameters["kernel_size"]
        p = self.parameters["pooling_size"]
        w  = self.parameters["encoder_width"]
        dw = self.parameters["width_increment"]
        return [
            Convolution2D(w * (dw ** i), (k,k), activation="relu", padding="same", use_bias=False),
            BN(),
            Dropout(self.parameters["encoder_dropout"]),
            MaxPooling2D((p,p)),
        ]

    def decoder_block(self,i):
        """Extend this method for Residual Nets"""
        k = self.parameters["kernel_size"]
        p = self.parameters["pooling_size"]
        w  = self.parameters["encoder_width"]
        dw = self.parameters["width_increment"]
        return [
            UpSampling2D((p,p)),
            Deconvolution2D(w * (dw ** i),(k,k), activation="relu",padding="same", use_bias=False),
            BN(),
            Dropout(self.parameters["decoder_dropout"]),
        ]

    def build_encoder(self,input_shape):
        if len(input_shape) == 2:
            reshape = [Reshape((*input_shape,1))] # monochrome image
        elif len(input_shape) == 3:
            reshape = []
        else:
            raise Exception(f"ConvolutionalEncoderMixin: unsupported shape {input_shape}")

        layers = [*reshape,
                  GaussianNoise(self.parameters["noise"]),
                  BN(),
        ]
        for i in range(self.parameters["encoder_depth"]-1):
            layers.extend(self.encoder_block(i))
        k = self.parameters["kernel_size"]
        layers.append(Convolution2D(self.parameters["P"], (k,k), padding="same"))

        self.conv_latent_space = self.output_shape(layers,[0,*input_shape])[1:] # H,W,C
        self.parameters["N"] = np.prod(self.conv_latent_space)
        print(f"latent space shape is {self.conv_latent_space} : {self.parameters['N']} propositions in total")
        return [*layers,
                # flatten,
                Reshape((self.parameters["N"],)),
                # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                # Conv2D does not set the tensor shape properly, and Flatten fails to work
                # during the build_aux phase.
                self.activation(),
        ]

    def build_decoder(self,input_shape):
        layers = [Reshape(self.conv_latent_space),
                  BN(),
        ]
        for i in range(self.parameters["encoder_depth"]-2, -1, -1):
            layers.extend(self.decoder_block(i))
        k = self.parameters["kernel_size"]
        layers.append(Deconvolution2D(3, (k,k), padding="same"))

        computed_input_shape = self.output_shape(layers,[0,*self.conv_latent_space])[1:]
        assert input_shape == computed_input_shape
        return layers


# Mixins ################################################################

class ZeroSuppressMixin:
    def _build(self,input_shape):
        super()._build(input_shape)

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["zerosuppress_delay"]):self.parameters["zerosuppress"],
        }, name="zerosuppress")
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
        if "locality_delay" not in self.parameters:
            self.parameters["locality_delay"] = 0.0
        if "locality" not in self.parameters:
            self.parameters["locality"] = 0.0
        self.locality_alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["locality_delay"]):self.parameters["locality"],
        }, name="locality")
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
    def _build_primary(self,input_shape):
        self.encoder_net = self.build_encoder(input_shape)
        self.decoder_net = self.build_decoder(input_shape)

        x = Input(shape=input_shape, name="autoencoder")
        z = Sequential(self.encoder_net)(x)
        y = Sequential(self.decoder_net)(z)

        if "loss" in self.parameters:
            self.loss = eval(self.parameters["loss"])
        else:
            self.loss = MSE

        if "eval" in self.parameters:
            e = eval(self.parameters["eval"])
            if e not in self.metrics:
                self.metrics.append(e)
            self.eval = e

        self.encoder     = Model(x, z)
        self.autoencoder = Model(x, y)
        self.net = self.autoencoder

    def _build_aux_primary(self,input_shape):
        # to be called after the training
        z2 = Input(shape=self.zdim(), name="autodecoder")
        y2 = Sequential(self.decoder_net)(z2)
        w2 = Sequential(self.encoder_net)(y2)
        self.decoder     = Model(z2, y2)
        self.autodecoder = Model(z2, w2)

    def plot(self,data,path,verbose=False):
        self.load()
        x = data
        z = self.encode(x)
        y = self.autoencode(x)

        xg = gaussian(x)
        xs = salt(x)
        xp = pepper(x)

        yg = self.autoencode(xg)
        ys = self.autoencode(xs)
        yp = self.autoencode(xp)

        dy  = ( y-x+1)/2
        dyg = (yg-x+1)/2
        dys = (ys-x+1)/2
        dyp = (yp-x+1)/2

        _z = squarify(z)

        _plot(path, (x, _z, y, dy, xg, yg, dyg, xs, ys, dys, xp, yp, dyp))
        return x,z,y

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

        M, N = self.parameters["M"], self.parameters["N"]

        _z   = squarify(z)
        _z2  = squarify(z2)
        _z2r = squarify(z2r)
        _z3  = squarify(z3)
        _z3r = squarify(z3r)

        _plot(path, (_z, x, _z2, _z2r, x2, x2r, _z3, _z3r, x3, x3r))
        return _z, x, _z2, _z2r

    def plot_variance(self,data,path,verbose=False):
        self.load()
        x = data
        samples = 100
        z = np.array([ np.round(self.encode(x)) for i in range(samples)])
        z = np.einsum("sbz->bsz",z)
        plot_grid(z, w=6, path=path, verbose=verbose)


class TransitionWrapper:
    def double_mode(self):
        pass
    def single_mode(self):
        pass
    def mode(self, single):
        pass

    def adaptively(self, fn, data, *args, **kwargs):
        if data.shape[1] == 2:
            return fn(data,*args,**kwargs)
        else:
            return fn(np.expand_dims(data,1).repeat(2, axis=1),*args,**kwargs)[:,0]

    def encode(self, data, *args, **kwargs):
        return self.adaptively(super().encode, data, *args, **kwargs)
    def decode(self, data, *args, **kwargs):
        return self.adaptively(super().decode, data, *args, **kwargs)
    def autoencode(self, data, *args, **kwargs):
        return self.adaptively(super().autoencode, data, *args, **kwargs)
    def autodecode(self, data, *args, **kwargs):
        return self.adaptively(super().autodecode, data, *args, **kwargs)


    def _build(self,input_shape):
        self.transition_input_shape = input_shape
        super()._build(input_shape[1:])

    def _build_aux(self,input_shape):
        self.transition_input_shape = input_shape
        super()._build_aux(input_shape[1:])

    def _build_primary(self,state_input_shape):
        # [batch, 2, ...] -> [batch, ...]
        self.encoder_net = self.build_encoder(state_input_shape)
        self.decoder_net = self.build_decoder(state_input_shape)

        x       = Input(shape=self.transition_input_shape, name="double_input")
        _, x_pre, x_suc = dapply(x, lambda x: x)
        z, z_pre, z_suc = dapply(x, Sequential(self.encoder_net))
        y, y_pre, y_suc = dapply(z, Sequential(self.decoder_net))

        self.encoder     = Model(x, z)
        self.autoencoder = Model(x, y)

        if "loss" in self.parameters:
            state_loss_fn = eval(self.parameters["loss"])
        else:
            state_loss_fn = MSE

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["main_delay"]):1,
        }, name="main")
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))
        def loss(x,y):
            return alpha.variable * (state_loss_fn(x_pre, y_pre) + state_loss_fn(x_suc, y_suc))
        self.loss = loss

        self.net = self.autoencoder

        if "eval" in self.parameters:
            e = eval(self.parameters["eval"])
            if e not in self.metrics:
                self.metrics.append(e)
            self.eval = e

        self.double_mode()
        return

    def _build_aux_primary(self,state_input_shape):

        z2       = Input(shape=(2,*self.zdim()), name="double_input_decoder")
        y2, _, _ = dapply(z2, Sequential(self.decoder_net))
        w2, _, _ = dapply(y2, Sequential(self.encoder_net))

        self.decoder     = Model(z2, y2)
        self.autodecoder = Model(z2, w2)
        return

    def dump_actions(self,pre,suc,**kwargs):
        def save(name,data):
            print("Saving to",self.local(name))
            with open(self.local(name), "wb") as f:
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
    def plot_transitions(self,data,path,verbose=False):
        import os.path
        basename, ext = os.path.splitext(path)
        pre_path = basename+"_pre"+ext
        suc_path = basename+"_suc"+ext

        x = data
        z = self.encode(x)
        y = self.autoencode(x)

        x_pre, x_suc = x[:,0,...], x[:,1,...]
        z_pre, z_suc = z[:,0,...], z[:,1,...]
        y_pre, y_suc_aae = y[:,0,...], y[:,1,...]
        y_suc = self.autoencode(x_suc) # run adaptively

        self.plot(x_pre,pre_path,verbose=verbose)
        self.plot(x_suc,suc_path,verbose=verbose)

        action    = self.encode_action(np.concatenate([z_pre,z_suc],axis=1))
        z_suc_aae = self.decode_action([z_pre, action])

        def diff(src,dst):
            return (dst - src + 1)/2

        _z_pre     = squarify(z_pre)
        _z_suc     = squarify(z_suc)
        _z_suc_aae = squarify(z_suc_aae)

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

    def add_metrics(self, x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae,):
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

        def mae_z2z3(true, pred):
            return K.mean(mae(K.round(z_suc), K.round(z_suc_aae)))

        def avg_z2(x, y):
            return K.mean(z_suc)
        def avg_z3(x, y):
            return K.mean(z_suc_aae)

        self.metrics.append(mse_x1y1)
        self.metrics.append(mse_x2y2)
        self.metrics.append(mse_x2y3)
        self.metrics.append(mse_y2y3)
        self.metrics.append(mae_z2z3)

        # self.metrics.append(avg_z2)
        self.metrics.append(avg_z3)

        return
    def build_action_fc_unit(self):
        return Sequential([
            Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
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

        x = self.net.input      # keras has a bug, we can"t make a new Input here
        _, x_pre, x_suc = dapply(x, lambda x: x)
        z, z_pre, z_suc = dapply(self.encoder.output,     lambda x: x)
        y, y_pre, y_suc = dapply(self.autoencoder.output, lambda x: x)

        if "stop_gradient" not in self.parameters:
            self.parameters["stop_gradient"] = False
        if self.parameters["stop_gradient"]:
            z_pre = wrap(z_pre, K.stop_gradient(z_pre))
            z_suc = wrap(z_suc, K.stop_gradient(z_suc))

        action    = self._action(z_pre,z_suc)
        z_suc_aae = self._apply(z_pre,z_suc,action)
        y_suc_aae = Sequential(self.decoder_net)(z_suc_aae)

        if "loss" in self.parameters:
            state_loss_fn = eval(self.parameters["loss"])
        else:
            state_loss_fn = MSE

        alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["successor_delay"]):1,
        }, name="successor")
        self.callbacks.append(LambdaCallback(on_epoch_end=alpha.update))
        self.net.add_loss(alpha.variable * K.mean(state_loss_fn(x_suc, y_suc_aae)))

        self.add_metrics(x_pre, x_suc, z_pre, z_suc, z_suc_aae, y_pre, y_suc, y_suc_aae)
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
            with open(self.local(name), "wb") as f:
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

        return


# action mapping variations

class DetActionMixin:
    "Deterministic mapping from a state pair to an action"
    def build_action_encoder(self):
        return [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            Sequential([
                        Dense(self.parameters["num_actions"]),
                        self.build_gs(N=1,
                                      M=self.parameters["num_actions"],
                                      offset=self.parameters["aae_delay"],),
            ]),
        ]
    def _action(self,z_pre,z_suc):
        self.action_encoder_net = self.build_action_encoder()
        return ConditionalSequential(self.action_encoder_net, z_pre, axis=-1)(z_suc)

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        N = self.parameters["N"]
        transition = Input(shape=(N*2,))
        pre2 = wrap(transition, transition[:,:N])
        suc2 = wrap(transition, transition[:,N:])
        self.action = Model(transition, ConditionalSequential(self.action_encoder_net, pre2, axis=-1)(suc2))
        return



# effect mapping variations

class DirectLossMixin:
    "Additional loss for the latent space successor prediction."
    def _build(self,input_shape):
        super()._build(input_shape)

        self.direct_alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["direct_delay"]):self.parameters["direct"],
        }, name="direct")
        self.callbacks.append(LambdaCallback(on_epoch_end=self.direct_alpha.update))

        return
    def apply_direct_loss(self,true,pred):
        dummy = Lambda(lambda x: x)
        if "direct_loss" not in self.parameters:
            self.parameters["direct_loss"] = "MAE"
        loss = K.mean(eval(self.parameters["direct_loss"])(true, pred))
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
            with open(self.local(name), "wb") as f:
                np.savetxt(f,data,"%d")

        N=pre.shape[1]
        data = np.concatenate([pre,suc],axis=1)
        actions = self.encode_action(data, **kwargs).round()
        # [B, 1, A]

        histogram = np.squeeze(actions.sum(axis=0,dtype=int))
        print(histogram)
        true_num_actions = np.count_nonzero(histogram)
        print(true_num_actions)
        all_labels = np.zeros((true_num_actions, actions.shape[1], actions.shape[2]), dtype=int)
        action_ids = np.where(histogram > 0)[0]
        for i, a in enumerate(action_ids):
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

        # extract the effects.
        # there were less efficient version 2 and 3, which uses the transition dataset.
        # this version does not require iterating over hte dataset --- merely twice over all actions.
        add_effect = self.decode_action([np.zeros((true_num_actions, N)),all_labels], **kwargs)
        del_effect = 1-self.decode_action([np.ones((true_num_actions, N)),all_labels], **kwargs)
        save("action_add4.csv",add_effect)
        save("action_del4.csv",del_effect)
        save("action_add4+ids.csv",np.concatenate((add_effect,action_ids.reshape([-1,1])), axis=1))
        save("action_del4+ids.csv",np.concatenate((del_effect,action_ids.reshape([-1,1])), axis=1))

        # extract the preconditions.
        # it is done by checking if a certain bit is always 0 or always 1.
        pos = []
        neg = []
        for a in action_ids:
            pre_a = pre[np.where(actions_byid == a)[0]]
            pos_a =   pre_a.min(axis=0,keepdims=True) # [1,C]
            neg_a = 1-pre_a.max(axis=0,keepdims=True) # [1,C]
            pos.append(pos_a)
            neg.append(neg_a)
        pos = np.concatenate(pos,axis=0) # [A,C]
        neg = np.concatenate(neg,axis=0) # [A,C]
        save("action_pos4.csv",pos)
        save("action_neg4.csv",neg)
        save("action_pos4+ids.csv",np.concatenate((pos,action_ids.reshape([-1,1])), axis=1))
        save("action_neg4+ids.csv",np.concatenate((neg,action_ids.reshape([-1,1])), axis=1))
        return

class ConditionalEffectMixin(BaseActionMixin,DirectLossMixin):
    "The effect depends on both the current state and the action labels -- Same as AAE in AAAI18."
    def _apply(self,z_pre,z_suc,action):

        self.action_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            Sequential([
                Dense(np.prod(self.zdim())),
                self.build_bc(),
                Reshape(self.zdim()),
            ])
        ]

        z_suc_aae = ConditionalSequential(self.action_decoder_net, z_pre, axis=1)(flatten(action))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        self.apply  = Model([pre2,act2], ConditionalSequential(self.action_decoder_net, pre2, axis=1)(flatten(act2)))
        return


class BoolMinMaxEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
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
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, K.minimum(1-del2, K.maximum(add2, pre2))))
        return


class BoolSmoothMinMaxEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary smooth min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
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
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, smooth_min(1-del2, smooth_max(add2, pre2))))
        return

class BoolAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are directly modeled as binary smooth min/max."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
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
        z_suc_aae = wrap(z_pre, self.build_bc()(z_pre - 0.5 + z_add - z_del))
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2     = Sequential(self.eff_decoder_net)(flatten(act2))
        add2     = wrap(eff2, eff2[...,0])
        del2     = wrap(eff2, eff2[...,1])
        self.apply  = Model([pre2,act2], wrap(pre2, self.build_bc()(pre2 - 0.5 + add2 - del2)))
        return

class NormalizedLogitAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique with batchnorm."
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            Sequential([
                Dense(np.prod(self.zdim()),use_bias=False),
                BN(),
            ])
        ]
        self.scaling = BN()

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = self.build_bc()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = self.build_bc()(lsuc2)
        self.apply  = Model([pre2,act2], suc2)
        return

class LogitAddEffectMixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique, but without batchnorm (s_t shifted from [0,1] to [-1/2,1/2].)"
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            Sequential([
                Dense(np.prod(self.zdim())),
            ])
        ]
        self.scaling = Lambda(lambda x: 2*x - 1) # same scale as the final batchnorm

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = self.build_bc()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = self.build_bc()(lsuc2)
        self.apply  = Model([pre2,act2], suc2)
        return

class LogitAddEffect2Mixin(ActionDumpMixin,BaseActionMixin,DirectLossMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique. Uses batchnorm in effect, but not in s_t (shifted from [0,1] to [-1/2,1/2].)"
    def _apply(self,z_pre,z_suc,action):

        self.eff_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            Sequential([
                Dense(np.prod(self.zdim())),
                BN(),
            ])
        ]
        self.scaling = Lambda(lambda x: 2*x - 1)

        l_eff     = Sequential(self.eff_decoder_net)(flatten(action))
        l_pre = self.scaling(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = self.build_bc()(l_suc_aae)
        z_suc_aae = self.apply_direct_loss(z_suc, z_suc_aae)
        return z_suc_aae

    def _build_aux(self, input_shape):
        super()._build_aux(input_shape)
        pre2 = Input(shape=self.zdim())
        act2 = Input(shape=(1,self.parameters["num_actions"],))
        eff2 = Sequential(self.eff_decoder_net)(flatten(act2))
        lpre2 = self.scaling(pre2)
        lsuc2 = add([lpre2,eff2])
        suc2 = self.build_bc()(lsuc2)
        self.apply  = Model([pre2,act2], suc2)
        return


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

class VanillaTransitionAE(              ZeroSuppressMixin, ConcreteLatentMixin, ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    pass

# earlier attempts to "sparcify" the transisions. No longer used
class HammingTransitionAE(HammingMixin, ZeroSuppressMixin, ConcreteLatentMixin, ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    pass
class CosineTransitionAE (CosineMixin,  ZeroSuppressMixin, ConcreteLatentMixin, ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    pass
class PoissonTransitionAE(PoissonMixin, ZeroSuppressMixin, ConcreteLatentMixin, ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    pass


# IJCAI2020 papers
class ConcreteDetConditionalEffectTransitionAE              (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, ConditionalEffectMixin,        ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Vanilla Space AE"""
    pass
class ConcreteDetBoolMinMaxEffectTransitionAE               (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolMinMaxEffectMixin,         ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Cube-Space AE with naive discrete effects (not BTL)"""
    pass
class ConcreteDetBoolSmoothMinMaxEffectTransitionAE         (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolSmoothMinMaxEffectMixin,   ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Cube-Space AE with naive discrete effects with smooth min/max"""
    pass
class ConcreteDetBoolAddEffectTransitionAE                  (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, BoolAddEffectMixin,            ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Cube-Space AE with naive discrete effects (not BTL). Effect is one-hot from add/del/nop"""
    pass
class ConcreteDetLogitAddEffectTransitionAE                 (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, LogitAddEffectMixin,           ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Cube-Space AE without BatchNorm"""
    pass
class ConcreteDetLogitAddEffect2TransitionAE                (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, LogitAddEffect2Mixin,          ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Cube-Space AE without BatchNorm for the current state but with BatchNorm for effects"""
    pass
class ConcreteDetNormalizedLogitAddEffectTransitionAE       (ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, NormalizedLogitAddEffectMixin, ConvolutionalEncoderMixin, TransitionWrapper, StateAE):
    """Final Cube-Space AE implementation"""
    pass



class FullyConvolutionalCubeSpaceAE(ZeroSuppressMixin, ConcreteLatentMixin, DetActionMixin, NormalizedLogitAddEffectMixin, FullyConvolutionalAEMixin, TransitionWrapper, StateAE):
    """Fully convolutional Cube-Space AE, cannot specify the latent space size (depends on the input size)"""
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
                          Dense(self.parameters["layer"],activation=self.parameters["activation"]),
                          Dropout(self.parameters["dropout"]),])
              for i in range(self.parameters["num_layers"]) ],
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
        K.set_value(self.c, self.parameters["c"])

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
            self.parameters["c"] = float(c)
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
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                Convolution2D(self.parameters["clayer"],(3,3),
                              activation=self.parameters["activation"],padding="same", use_bias=False),
                Dropout(self.parameters["dropout"]),
                BN(),
                MaxPooling2D((2,2)),
                flatten,]

    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(self.parameters["layer"], activation="relu", use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
            Dense(data_dim, activation="sigmoid"),
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
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                    Dense(self.parameters["N"]*self.parameters["M"]),
                    self.build_gs(),
            ]),
        ]

    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                Dense(data_dim),
                self.build_bc(),
                Reshape(input_shape),]),]

    def _build(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters["M"], self.parameters["N"]

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
        M, N = self.parameters["M"], self.parameters["N"]
        return self.encode(data,**kwargs)[1]

    def report(self,train_data,
               epoch=200,batch_size=1000,optimizer=Adam(0.001),
               test_data=None,
               train_data_to=None,
               test_data_to=None,):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {"verbose":0,"batch_size":batch_size}
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

        x_pre, x_suc = squarify(x[:,:dim]), squarify(x[:,dim:])
        y_pre, y_suc = squarify(y[:,:dim]), squarify(y[:,dim:])
        by_pre, by_suc = squarify(by[:,:dim]), squarify(by[:,dim:])
        y_suc_r, by_suc_r = y_suc.round(), by_suc.round()

        if sae:
            x_pre_im, x_suc_im = sae.decode(x[:,:dim]), sae.decode(x[:,dim:])
            y_pre_im, y_suc_im = sae.decode(y[:,:dim]), sae.decode(y[:,dim:])
            by_pre_im, by_suc_im = sae.decode(by[:,:dim]), sae.decode(by[:,dim:])
            y_suc_r_im, by_suc_r_im = sae.decode(y[:,dim:].round()), sae.decode(by[:,dim:].round())
            _plot(self.local(path),
                  (x_pre_im, x_suc_im, squarify(np.squeeze(z)),
                   y_pre_im, y_suc_im, y_suc_r_im, squarify(np.squeeze(b)),
                   by_pre_im, by_suc_im, by_suc_r_im))
        else:
            _plot(self.local(path),
                  (x_pre, x_suc, squarify(np.squeeze(z)),
                   y_pre, y_suc, y_suc_r, squarify(np.squeeze(b)),
                   by_pre, by_suc, by_suc_r))
        return x,z,y,b,by

class CubeActionAE(ActionAE):
    """AAE with cube-like structure, developped for a compariason purpose."""
    def build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [
            *[
                Sequential([
                    Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                for i in range(self.parameters["aae_depth"]-1)
            ],
            Sequential([
                Dense(data_dim,use_bias=False),
                BN(),
            ]),
        ]

    def _build(self,input_shape):

        dim = np.prod(input_shape) // 2
        print("{} latent bits".format(dim))
        M, N = self.parameters["M"], self.parameters["N"]

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
        suc_reconstruction = self.build_bc()(l_suc)

        y = Concatenate(axis=1)([pre,suc_reconstruction])

        action2 = Input(shape=(N,M))
        pre2    = Input(shape=(dim,))
        l_pre2 = scaling(pre2)
        l_eff2 = Sequential(_decoder)(flatten(action2))
        l_suc2 = add([l_eff2,l_pre2])
        suc_reconstruction2 = self.build_bc()(l_suc2)
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
                              Dense(self.parameters["layer"],activation=self.parameters["activation"]),
                              Dropout(self.parameters["dropout"]),])
              for i in range(self.parameters["num_layers"]) ],
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


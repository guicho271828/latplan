#!/usr/bin/env python3

"""
Model classes for latplan.
"""

import json
import numpy as np
from keras.layers import *
from keras.layers.normalization import BatchNormalization as BN
from keras.models import Model
from keras.optimizers import Adam
import keras.backend.tensorflow_backend as K
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback, CallbackList, ReduceLROnPlateau
import tensorflow as tf
from .util.noise import gaussian
from .util.distances import *
from .util.layers    import *
from .util.perminv   import *
from .util.tuning    import InvalidHyperparameterError
from .util.plot      import plot_grid, squarify
from .util           import ensure_list, NpEncoder, curry
from .util.stacktrace import print_object
from .mixins.latent          import *
from .mixins.encoder_decoder import *
from .mixins.locality        import *
from .mixins.output          import *
from .network                import Network

# Style guide
# * Each section is divided by 3 newlines
# * Each class is separated by 2 newline (so that hideshow will separate them by 1 newline when folded)

# utilities ####################################################################

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



# Network mixins ###############################################################

class AE(Network):
    """Autoencoder class. Supports SAVE and LOAD, as well as REPORT methods.
Additionally, provides ENCODE / DECODE / AUTOENCODE / AUTODECODE methods.
The latter two are used for verifying the performance of the AE.
"""
    def report(self,train_data,
               test_data=None,
               train_data_to=None,
               test_data_to=None,
               batch_size=400,
               **kwargs):
        test_data     = train_data if test_data is None else test_data
        train_data_to = train_data if train_data_to is None else train_data_to
        test_data_to  = test_data  if test_data_to is None else test_data_to
        opts = {"verbose":0,"batch_size":batch_size}

        performance = {}

        optimizer  = self.parameters["optimizer"]
        lr         = self.parameters["lr"]
        clipnorm   = self.parameters["clipnorm"]
        def make_optimizer(net):
            return getattr(keras.optimizers,optimizer)(
                lr,
                clipnorm=clipnorm
                # clipvalue=clipvalue,
            )

        self.optimizers = list(map(make_optimizer, self.nets))
        self.compile(self.optimizers)

        def reg(query, data, d={}):
            if len(query) == 1:
                d[query[0]] = data
                return d
            if query[0] not in d:
                d[query[0]] = {}
            reg(query[1:],data,d[query[0]])
            return d

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

    def build_gs(self,N=None,M=None,
                 **kwargs):
        # python methods cannot use self in the
        # default values, because python sucks
        assert (N is not None) or (M is not None)
        def fn(max_temperature = self.parameters["max_temperature"],
               min_temperature = self.parameters["min_temperature"],
               annealing_start = self.parameters["gs_annealing_start"],
               annealing_end   = self.parameters["gs_annealing_end"],
               train_noise     = self.parameters["train_noise"],
               train_hard      = self.parameters["train_hard"],
               test_noise      = self.parameters["test_noise"],
               test_hard       = self.parameters["test_hard"]):
            gs = GumbelSoftmax(
                N,M,min_temperature,max_temperature,
                annealing_start,
                annealing_end,
                train_noise = train_noise,
                train_hard  = train_hard,
                test_noise  = test_noise,
                test_hard   = test_hard)
            self.callbacks.append(LambdaCallback(on_epoch_end=gs.update))
            self.add_metric("tau",gs.variable)
            return gs

        return fn(**kwargs)

    def build_bc(self,
                 **kwargs):
        # python methods cannot use self in the
        # default values, because python sucks
        def fn(max_temperature = self.parameters["max_temperature"],
               min_temperature = self.parameters["min_temperature"],
               annealing_start = self.parameters["gs_annealing_start"],
               annealing_end   = self.parameters["gs_annealing_end"],
               train_noise     = self.parameters["train_noise"],
               train_hard      = self.parameters["train_hard"],
               test_noise      = self.parameters["test_noise"],
               test_hard       = self.parameters["test_hard"]):
            bc = BinaryConcrete(
                min_temperature,max_temperature,
                annealing_start,
                annealing_end,
                train_noise = train_noise,
                train_hard  = train_hard,
                test_noise  = test_noise,
                test_hard   = test_hard)
            self.callbacks.append(LambdaCallback(on_epoch_end=bc.update))
            return bc

        return fn(**kwargs)



# Mixins #######################################################################

class ZeroSuppressMixin:
    def _build_around(self,input_shape):
        super()._build_around(input_shape)

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
    def _build_around(self,input_shape):
        super()._build_around(input_shape)

        # check all hyperparameters and ensure that the earlystop does not activate until all
        # delayed loss epoch kicks in
        max_delay = 0.0
        for key in self.parameters:
            if "delay" in key:
                max_delay = max(max_delay, self.parameters[key])
        earlystop_delay = max_delay + 0.1

        # self.callbacks.append(
        #     ChangeEarlyStopping(
        #         epoch_start=self.parameters["epoch"]*earlystop_delay,
        #         verbose=1,))

        # self.callbacks.append(
        #     LinearEarlyStopping(
        #         self.parameters["epoch"],
        #         epoch_start = self.parameters["epoch"]*earlystop_delay,
        #         target_value = self.parameters["current_best"],
        #         verbose     = 1,))

        # self.callbacks.append(
        #     ExplosionEarlyStopping(
        #         self.parameters["epoch"],
        #         epoch_start=10,
        #         verbose     = 1,))

        pass



# State Auto Encoder ###########################################################

# AAAI18 paper
class StateAE(ConcreteLatentMixin, EarlyStopMixin, FullConnectedMixin, AE):
    """An AE whose latent layer is BinaryConcrete.
Fully connected layers only, no convolutions.
Note: references to self.parameters[key] are all hyperparameters."""
    def _build_primary(self,input_shape):
        self.loss = self.output.loss

        x = Input(shape=input_shape, name="autoencoder")
        z = self._encode(x)
        y = self._decode(z)

        self.encoder     = Model(x, z)
        self.autoencoder = Model(x, y)
        self.net = self.autoencoder

    def _build_aux_primary(self,input_shape):
        # to be called after the training
        z2 = Input(shape=self.zdim(), name="autodecoder")
        y2 = self._decode(z2)
        w2 = self._encode(y2)
        self.decoder     = Model(z2, y2)
        self.autodecoder = Model(z2, w2)

    def render(self,x):
        return self.output.render(x)

    def plot(self,x,path,verbose=False,epoch=None):
        self.load()
        z = self.encode(x)
        y = self.autoencode(x)

        xg = gaussian(x)

        yg = self.autoencode(xg)

        x = self.render(x)
        y = self.render(y)
        xg = self.render(xg)
        yg = self.render(yg)

        dy  = ( y-x+1)/2
        dyg = (yg-x+1)/2

        _z = squarify(z)

        self._plot(path, (x, _z, y, dy, xg, yg, dyg),epoch=epoch)
        return x,z,y

    def plot_cycle(self,x1,path,cycles=3,verbose=False,epoch=None):
        self.load()
        B, *_ = x1.shape

        def diff(src,dst):
            return (dst - src + 1)/2

        rx1 = self.render(x1)
        rz1 = squarify(self.encode(x1).reshape((B,-1)))
        print("rx1.min()",rx1.min(),"rx1.max()",rx1.max())
        print("rz1.min()",rz1.min(),"rz1.max()",rz1.max())

        results_x = [rx1]
        results_z = [rz1]
        results_dx = []
        results_dz = []

        for i in range(cycles):
            x2  = self.autoencode(x1)
            rx2 = self.render(x2)
            rz2 = squarify(self.encode(x2).reshape((B,-1)))
            print("rx2.min()",rx2.min(),"rx2.max()",rx2.max())
            print("rz2.min()",rz2.min(),"rz2.max()",rz2.max())
            results_x.append(rx2)
            results_z.append(rz2)
            results_dx.append(diff(rx1,rx2))
            results_dz.append(diff(rz1,rz2))
            x1 = x2
            rx1 = rx2
            rz1 = rz2

        import os.path
        name, ext = os.path.splitext(path)
        self._plot(name+"_x"+ext, results_x, epoch=epoch)
        self._plot(name+"_z"+ext, results_z, epoch=epoch)
        self._plot(name+"_dx"+ext, results_dx, epoch=epoch)
        self._plot(name+"_dz"+ext, results_dz, epoch=epoch)
        return

    def plot_plan(self,z,path,verbose=False):
        "Plot a sequence of states horizontally."
        self.load()
        y = self.decode(z)
        y = self.render(y)
        y = [ r for r in y ]
        plot_grid(y, w=8, path=path, verbose=True)
        return


    def dump_actions(self,*args,**kwargs):
        """Is here so that SAE and TAE has the consistent interface"""
        pass


    def _report(self,test_both,**opts):

        from .util.np_distances import mse, mae

        def metrics(data):
            return { k:v for k,v in zip(self.net.metrics_names,
                                        ensure_list(self.net.evaluate(data, data, **opts)))}

        test_both(["metrics"], metrics)

        # evaluate the current state reconstruction loss.
        # This is using self.autoencoder rather than self.net, therefore the second reconstruction is through the SAE, not AAE
        test_both(["sae","mse","vanilla"],  lambda data: mse(data, self.autoencoder.predict(data,          **opts)))
        test_both(["sae","mse","gaussian"], lambda data: mse(data, self.autoencoder.predict(gaussian(data),**opts)))

        test_both(["sae","activation"], lambda data: float(self.encode(data,**opts).mean()))

        # UGLY! The following functions are added because this code path should handle both the single-state case
        # and the double-state (transition) case.
        def batch_amax_mod(z):
            # this is latent space vectors, so it is either [batch,2,N] or [batch,N]
            if z.ndim == 2:
                return np.amax(z,axis=0)
            # the input is a latent transition (z0, z1)
            if z.ndim == 3:
                return np.amax(z,axis=(0,1))
            assert False
        def batch_amin_mod(z):
            # this is latent space vectors, so it is either [batch,2,N] or [batch,N]
            if z.ndim == 2:
                return np.amin(z,axis=0)
            # the input is a latent transition (z0, z1)
            if z.ndim == 3:
                return np.amin(z,axis=(0,1))
            assert False

        test_both(["sae","ever_1"],     lambda data: float(np.sum(batch_amax_mod(self.encode(data,**opts)))))
        test_both(["sae","ever_0"],     lambda data: float(np.sum(1-batch_amin_mod(self.encode(data,**opts)))))
        test_both(["sae","effective"],  lambda data: float(np.sum((1-batch_amin_mod(self.encode(data,**opts)))*batch_amax_mod(self.encode(data,**opts)))))

        def latent_variance_noise(data,noise):
            encoded = [self.encode(noise(data),**opts).round() for i in range(10)]
            var = np.var(encoded,axis=0)
            return {
                "max":float(np.amax(var)),
                "min":float(np.amin(var)),
                "mean":float(np.mean(var)),
                "median":float(np.median(var)),
            }

        test_both(["sae","variance","gaussian"], lambda data: latent_variance_noise(data,gaussian))

        def cycle_consistency(data):
            x0 = data
            z0 = self.encoder.predict(x0,**opts)
            x1 = x0
            z1 = z0
            xs01 = []            # does x move away from the initial x?
            xs12 = []            # does x converge?
            zs01 = []            # does z move away from the initial z?
            zs12 = []            # does z converge?
            for i in range(10):
                xs01.append(mse(x0,x1))
                zs01.append(mae(z0,z1))
                x2 = self.decoder.predict(z1,**opts)
                z2 = self.encoder.predict(x2,**opts)
                xs12.append(mse(x1,x2))
                zs12.append(mae(z1,z2))
                z1 = z2
                x1 = x2

            # noisy input
            x3 = gaussian(x0)
            z3 = self.encoder.predict(x3,**opts)
            xs13 = []           # does x with noise converge to the same value (x1)?
            xs34 = []           # does x converge?
            zs13 = []
            zs34 = []
            for i in range(10):
                xs13.append(mse(x1,x3))
                zs13.append(mae(z1,z3))
                x4 = self.decoder.predict(z3,**opts)
                z4 = self.encoder.predict(x4,**opts)
                xs34.append(mse(x3,x4))
                zs34.append(mae(z3,z4))
                z3 = z4
                x3 = x4

            return {"xs01":xs01,
                    "xs12":xs12,
                    "xs13":xs13,
                    "xs34":xs34,
                    "zs01":zs01,
                    "zs12":zs12,
                    "zs13":zs13,
                    "zs34":zs34}

        test_both(["cycle_consistency"], cycle_consistency)

        return



# Transition Auto Encoder ######################################################

# Warning!!! Action and effect mapping mixins are used before TransitionWrapper in the precedence list.
# TransitionWrapper removes the first element (which should be 2) in the input_shape.
# If your forgot this and use e.g. input_shape[0] trying to obtain the size of the first dimension of the state,
# you are screwed! You instead get 2.

class TransitionWrapper:
    """A wrapper over SAEs which makes it able to handle 2 states at once. This does not imply it learns an action."""
    def double_mode(self):
        pass
    def single_mode(self):
        pass
    def mode(self, single):
        pass

    def adaptively(self, fn, data, *args, **kwargs):
        try:
            return fn(data,*args,**kwargs)
        except ValueError:
            # not transitions
            return fn(np.expand_dims(data,1).repeat(2, axis=1),*args,**kwargs)[:,0]

    def encode(self, data, *args, **kwargs):
        return self.adaptively(super().encode, data, *args, **kwargs)
    def decode(self, data, *args, **kwargs):
        return self.adaptively(super().decode, data, *args, **kwargs)
    def autoencode(self, data, *args, **kwargs):
        return self.adaptively(super().autoencode, data, *args, **kwargs)
    def autodecode(self, data, *args, **kwargs):
        return self.adaptively(super().autodecode, data, *args, **kwargs)


    def _build_around(self,input_shape):
        super()._build_around(input_shape[1:])

    def _build_aux_around(self,input_shape):
        super()._build_aux_around(input_shape[1:])

    def _build_primary(self,input_shape):
        # [batch, 2, ...] -> [batch, ...]

        x       = Input(shape=(2,*input_shape))
        _, x_pre, x_suc = dapply(x)
        z, z_pre, z_suc = dapply(x, self._encode)
        y, y_pre, y_suc = dapply(z, self._decode)

        self.encoder     = Model(x, z)
        self.autoencoder = Model(x, y)

        state_loss_fn = self.output.loss

        def loss(x,y):
            return state_loss_fn(x_pre, y_pre) + state_loss_fn(x_suc, y_suc)
        self.loss = loss

        self.net = self.autoencoder

        self.double_mode()
        return

    def _build_aux_primary(self,input_shape):

        z2       = Input(shape=(2,*self.zdim()), name="double_input_decoder")
        y2, _, _ = dapply(z2, self._decode)
        w2, _, _ = dapply(y2, self._encode)

        self.decoder     = Model(z2, y2)
        self.autodecoder = Model(z2, w2)
        return

    def dump_actions(self,transitions,**kwargs):
        """Since TransitionWrapper may not have action discovery (AAE), it just saves a set of concatenated transitions"""
        transitions_z = self.encode(transitions,**kwargs)
        pre = transitions_z[:,0,...]
        suc = transitions_z[:,1,...]
        data = np.concatenate([pre,suc],axis=1)
        self.save_array("actions.csv", data)
        return


class BaseActionMixin:
    """An abstract mixin for learning action labels/symbols. This may or may not have interpretable effect learning."""
    def encode_action(self,data,**kwargs):
        return self.action.predict(data,**kwargs)
    def apply(self,data,**kwargs):
        return self.applier.predict(data,**kwargs)
    def regress(self,data,**kwargs):
        return self.regressor.predict(data,**kwargs)

    def build_action_fc_unit(self):
        return Sequential([
            Dense(self.parameters["aae_width"], activation=self.parameters["aae_activation"], use_bias=False),
            BN(),
            Dropout(self.parameters["dropout"]),
        ])

    def apply_direct_loss(self,true,pred):
        if true is None:
            return pred
        dummy = Lambda(lambda x: x)
        loss  = K.mean(eval(self.parameters["direct_loss"])(true, pred))
        loss1 = K.mean(SymmetricBCE(true, pred)) * self.parameters["direct_eval"]
        # direct loss should be treated as the real loss
        dummy.add_loss(K.in_train_phase(loss * self.direct_alpha.variable, loss1))
        def direct(x,y):
            return loss1
        self.metrics.append(direct)
        return dummy(pred)

    def _dump_actions_prologue(self,pre,suc,**kwargs):
        """Compute and return a matrix, whose each row is a one-hot vector.
It contans only as many rows as the available actions. Unused actions are removed."""
        N=pre.shape[1]
        data = np.concatenate([pre,suc],axis=1)
        actions = self.encode_action([pre,suc], **kwargs).round()

        histogram = np.squeeze(actions.sum(axis=0,dtype=int))
        print(histogram)
        true_num_actions = np.count_nonzero(histogram)
        print(true_num_actions)
        all_labels = np.zeros((true_num_actions, actions.shape[1], actions.shape[2]), dtype=int)
        action_ids = np.where(histogram > 0)[0]
        for i, a in enumerate(action_ids):
            all_labels[i][0][a] = 1

        return pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids

    def edim(self):
        "Returns the effect size. In a grounded representation, this is same as the size of the propositional state."
        return self.zdim()

    def dump_effects(self, pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids):
        pass
    def dump_preconditions(self, pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids):
        pass
    def dump_actions(self,transitions,**kwargs):
        transitions_z = self.encode(transitions,**kwargs)
        pre = transitions_z[:,0,...]
        suc = transitions_z[:,1,...]
        pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**kwargs)
        self.save_array("available_actions.csv", action_ids)
        self.dump_effects      (pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids, **kwargs)
        self.dump_preconditions(pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids, **kwargs)
    pass



# action mapping variations
# Note: There were two more variations (see: old/precondition-networks.py), but now they do not make sense (JAIR)

class DetActionMixin:
    "Deterministic mapping from a state pair to an action"
    def _build_around(self, input_shape):
        self.action_encoder_net = [
            (Densify if self.parameters["densify"] else Sequential)
            ([self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)]),
            Sequential([
                Dense(self.parameters["A"]),
                self.build_gs(N=1,
                              M=self.parameters["A"],),
            ]),
        ]
        super()._build_around(input_shape)

    def adim(self):
        return (1,self.parameters["A"])

    def _action(self,p_pre,p_suc):
        return Sequential(self.action_encoder_net)(concatenate([p_pre,p_suc],axis=-1))

    def _build_aux_around(self, input_shape):
        super()._build_aux_around(input_shape)
        z_pre = Input(shape=self.edim())
        z_suc = Input(shape=self.edim())
        action = self._action(z_pre, z_suc)
        self.action = Model([z_pre, z_suc], action)

        return

    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)

        def encode_and_split(data):
            z     = self.encode(data)
            pre, suc = z[:,0,...], z[:,1,...]
            return [pre, suc]

        def true_num_actions(data):
            actions = self.encode_action(encode_and_split(data), **opts).round()
            histogram = np.squeeze(actions.sum(axis=0,dtype=int))
            true_num_actions = np.count_nonzero(histogram)
            return true_num_actions

        test_both(["action","true_num_actions"], true_num_actions)

        return


# effect mapping variations

class EffectMixin:
    def _build_aux_around(self, input_shape):
        super()._build_aux_around(input_shape)

        z      = Input(shape=self.edim())
        action = Input(shape=self.adim())
        self.applier = Model([z,action], self._apply(z, action))

        return

    def dump_effects(self, pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids, **kwargs):

        # extract the effects.
        # there were less efficient version 2 and 3, which uses the transition dataset.
        # this version does not require iterating over hte dataset --- merely twice over all actions.
        add_effect = self.apply([np.zeros((true_num_actions, *self.edim())),all_labels], **kwargs)
        del_effect = 1-self.apply([np.ones((true_num_actions, *self.edim())),all_labels], **kwargs)
        self.save_array("action_add4.csv",add_effect)
        self.save_array("action_del4.csv",del_effect)
        self.save_array("action_add4+ids.csv",np.concatenate((add_effect,action_ids.reshape([-1,1])), axis=1))
        self.save_array("action_del4+ids.csv",np.concatenate((del_effect,action_ids.reshape([-1,1])), axis=1))

        return

    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)

        from .util.np_distances import mse, mae

        # note: This is using self.net assuming it will return y0 and y2. It should not be self.autoencoder, which returns y0 and y1.
        test_both(["x1y2","mse","vanilla"],  lambda data: mse(data[:,1,...], self.net.predict(data,          **opts)[:,1,...]))
        test_both(["x1y2","mse","gaussian"], lambda data: mse(data[:,1,...], self.net.predict(gaussian(data),**opts)[:,1,...]))

        def encode_and_split(data):
            z     = self.encode(data)
            z_pre, z_suc = z[:,0,...], z[:,1,...]
            actions = self.encode_action([z_pre, z_suc], **opts).round()
            z_suc_aae = self.apply([z_pre,actions], **opts)
            return z_pre, z_suc, z_suc_aae

        def z_mae(data):
            z_pre, z_suc, z_suc_aae = encode_and_split(data)
            return mae(z_suc, z_suc_aae)

        test_both(["z1z2","mae","vanilla"], z_mae)
        test_both(["z1z2","mae","gaussian"],lambda data: z_mae(gaussian(data)))

        def z_match_prob(data):
            z_pre, z_suc, z_suc_aae = encode_and_split(data)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.prod(np.mean(z_match,axis=0))

        test_both(["z1z2","match_prob","vanilla"], z_match_prob)
        test_both(["z1z2","match_prob","gaussian"],lambda data: z_match_prob(gaussian(data)))

        def z_allmatch_prob(data):
            z_pre, z_suc, z_suc_aae = encode_and_split(data)
            z_match   = 1-np.abs(z_suc_aae-z_suc)
            return np.mean(np.prod(z_match,axis=1))

        test_both(["z1z2","allmatch_prob","vanilla"], z_allmatch_prob)
        test_both(["z1z2","allmatch_prob","gaussian"],lambda data: z_allmatch_prob(gaussian(data)))

        def z_xor(data):
            z     = self.encode(data)
            pre = z[:,0,...]
            suc = z[:,1,...]
            pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**opts)

            add_effect = self.apply([np.zeros((true_num_actions, *self.edim())),all_labels], **opts)
            del_effect = 1-self.apply([np.ones((true_num_actions, *self.edim())),all_labels], **opts)
            xor_effect = add_effect * del_effect # 1 when both add and del are 1
            # now this is [A, N] matrix.
            return np.mean(np.sum(xor_effect,axis=1))

        test_both(["z1z2","xor"], z_xor)

        def compiled_action(data):
            z     = self.encode(data)
            pre = z[:,0,...]
            suc = z[:,1,...]
            pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**opts)

            add_effect = self.apply([np.zeros((true_num_actions, *self.edim())),all_labels], **opts)
            del_effect = 1-self.apply([np.ones((true_num_actions, *self.edim())),all_labels], **opts)
            xor_effect = add_effect * del_effect # 1 when both add and del are 1
            # now this is [A, N] matrix.
            xor_number_per_action = np.sum(xor_effect,axis=1)
            compiled_actions_per_action = 2 ** xor_number_per_action
            total_compiled_action = np.sum(compiled_actions_per_action)
            return total_compiled_action

        test_both(["z1z2","compiled_action"], compiled_action)

        def kl_a_z(data):
            metrics = { k:v for k,v in zip(self.net.metrics_names,
                                           ensure_list(self.net.evaluate(data, data, **opts)))}
            return metrics["kl_a_z0"]

        test_both(["kl_a_z0"], kl_a_z)

        return

    pass


class ConditionalEffectMixin:
    """The effect depends on both the current state and the action labels -- Same as AAE in AAAI18.
It overwrites dump_actions because effects/preconditions must be learned separately."""
    def _build_primary(self,input_shape):
        self.action_decoder_net = [
            *[self.build_action_fc_unit() for i in range(self.parameters["aae_depth"]-1)],
            # this must be wrapped like this for ConditionalSequential
            Sequential([
                Dense(np.prod(self.edim())),
                self.build_bc(),
                Reshape(self.edim()),
            ])
        ]
        super()._build_primary(input_shape)

    def _apply(self,z_pre,action):
        z_suc_aae = ConditionalSequential(self.action_decoder_net, z_pre, axis=1)(MyFlatten()(action))
        return z_suc_aae

    def dump_actions(self,transitions,**kwargs):
        transitions_z = self.encode(transitions,**kwargs)
        pre = transitions_z[:,0,...]
        suc = transitions_z[:,1,...]

        pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**kwargs)
        self.save_array("available_actions.csv", action_ids)

        A = self.parameters["A"]
        def to_id(actions):
            return (actions * np.arange(A)).sum(axis=-1,dtype=int)
        actions_byid = to_id(actions)
        data_byid = np.concatenate((data,actions_byid), axis=1)
        data_aae = np.concatenate([pre,self.apply([pre,actions], **kwargs)], axis=1)
        data_aae_byid = np.concatenate((data_aae,actions_byid), axis=1)
        self.save_array("actions.csv", data)
        self.save_array("actions+ids.csv", data_byid)
        self.save_array("actions_aae.csv", data_aae)
        self.save_array("actions_aae+ids.csv", data_aae_byid)
        self.save_array("actions_both.csv", np.concatenate([data,data_aae], axis=0))
        self.save_array("actions_both+ids.csv", np.concatenate([data_byid,data_aae_byid], axis=0))

        return



class CategoricalEffectMixin:
    def _build_primary(self,input_shape):
        self.eff_decoder_net = [
            MyFlatten(),
            Dense(np.prod(self.edim())*3),
            Reshape((*self.edim(),3)),
            self.build_gs(),
            Reshape((*self.edim(),3)),
        ]
        super()._build_primary(input_shape)


class BoolMinMaxEffectMixin(CategoricalEffectMixin):
    "A naive effect learning method. Add/delete effects are directly modeled as binary min/max."
    def _apply(self,z_pre,action):
        z_eff     = Sequential(self.eff_decoder_net)(action)
        z_add     = wrap(z_eff, z_eff[...,0])
        z_del     = wrap(z_eff, z_eff[...,1])
        z_suc_aae = wrap(z_pre, K.minimum(1-z_del, K.maximum(z_add, z_pre)))
        return z_suc_aae


class BoolSmoothMinMaxEffectMixin(CategoricalEffectMixin):
    "A naive effect learning method. Add/delete effects are directly modeled as binary smooth min/max."
    def _apply(self,z_pre,action):
        z_eff     = Sequential(self.eff_decoder_net)(action)
        z_add     = wrap(z_eff, z_eff[...,0])
        z_del     = wrap(z_eff, z_eff[...,1])
        z_suc_aae = wrap(z_pre, smooth_min(1-z_del, smooth_max(z_add, z_pre)))
        return z_suc_aae



class LogitEffectMixin:
    def _build_primary(self,input_shape):
        self.eff_decoder_net = [
            MyFlatten(),
            Dense(np.prod(self.edim()),use_bias=False,kernel_regularizer=self.parameters["eff_regularizer"]),
            *([BN()] if self.btl_eff_batchnorm else [])
        ]
        if self.btl_pre_batchnorm:
            self.scaling_pre = BN()
        else:
            self.scaling_pre = Lambda(lambda x: 2*x - 1) # same scale as the final batchnorm
        super()._build_primary(input_shape)

    def _apply(self,z_pre,action):
        l_eff     = Sequential(self.eff_decoder_net)(action)
        l_pre     = self.scaling_pre(z_pre)
        l_suc_aae = add([l_pre,l_eff])
        z_suc_aae = self.build_bc()(l_suc_aae)
        return z_suc_aae


class NormalizedLogitAddEffectMixin(LogitEffectMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique with batchnorm."
    btl_eff_batchnorm = True
    btl_pre_batchnorm = True
    pass


class LogitAddEffectMixin(LogitEffectMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique. Uses batchnorm in s_t, but not in effect (shifted from [0,1] to [-1/2,1/2].)"
    btl_eff_batchnorm = False
    btl_pre_batchnorm = True
    pass


class LogitAddEffect2Mixin(LogitEffectMixin):
    "The effect depends only on the action labels. Add/delete effects are implicitly modeled by back2logit technique. Uses batchnorm in effect, but not in s_t (shifted from [0,1] to [-1/2,1/2].)"
    btl_eff_batchnorm = True
    btl_pre_batchnorm = False
    pass



# precondition mapping variations

class AddHocPreconditionMixin:
    def dump_preconditions(self, pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids, **kwargs):
        A = self.parameters["A"]
        # extract the preconditions.
        # it is done by checking if a certain bit is always 0 or always 1.
        def to_id(actions):
            return (actions * np.arange(A)).sum(axis=-1,dtype=int)
        actions_byid = to_id(actions)
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
        self.save_array("action_pos4.csv",pos)
        self.save_array("action_neg4.csv",neg)
        self.save_array("action_pos4+ids.csv",np.concatenate((pos,action_ids.reshape([-1,1])), axis=1))
        self.save_array("action_neg4+ids.csv",np.concatenate((neg,action_ids.reshape([-1,1])), axis=1))
        return


class PreconditionMixin:
    def _build_aux_around(self, input_shape):
        super()._build_aux_around(input_shape)

        z      = Input(shape=self.edim())
        action = Input(shape=self.adim())
        self.regressor  = Model([z,action], self._regress(z, action))

        return

    def dump_preconditions(self, pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids, **kwargs):

        # extract the preconditions with deterministic regression.
        pos_precondition = self.regress([np.zeros((true_num_actions, *self.edim())),all_labels], **kwargs)
        neg_precondition = 1-self.regress([np.ones((true_num_actions, *self.edim())),all_labels], **kwargs)
        self.save_array("action_pos4.csv",pos_precondition)
        self.save_array("action_neg4.csv",neg_precondition)
        self.save_array("action_pos4+ids.csv",np.concatenate((pos_precondition,action_ids.reshape([-1,1])), axis=1))
        self.save_array("action_neg4+ids.csv",np.concatenate((neg_precondition,action_ids.reshape([-1,1])), axis=1))
        return

    def _report(self,test_both,**opts):
        super()._report(test_both,**opts)

        from .util.np_distances import mse, mae

        # note: This is using self.net assuming it will return y3 and y2. It should not be self.autoencoder, which returns y0 and y1.
        test_both(["x0y3","mse","vanilla"],  lambda data: mse(data[:,0,...], self.net.predict(data,          **opts)[:,0,...]))
        test_both(["x0y3","mse","gaussian"], lambda data: mse(data[:,0,...], self.net.predict(gaussian(data),**opts)[:,0,...]))

        def encode_and_split(data):
            z     = self.encode(data)
            z_pre, z_suc = z[:,0,...], z[:,1,...]
            actions = self.encode_action([z_pre, z_suc], **opts).round()
            z_pre_aae = self.regress([z_suc,actions], **opts)
            return z_pre, z_suc, z_pre_aae

        def z_mae(data):
            z_pre, z_suc, z_pre_aae = encode_and_split(data)
            return mae(z_pre, z_pre_aae)

        test_both(["z0z3","mae","vanilla"], z_mae)
        test_both(["z0z3","mae","gaussian"],lambda data: z_mae(gaussian(data)))

        def z_match_prob(data):
            z_pre, z_suc, z_pre_aae = encode_and_split(data)
            z_match   = 1-np.abs(z_pre_aae-z_pre)
            return np.prod(np.mean(z_match,axis=0))

        test_both(["z0z3","match_prob","vanilla"], z_match_prob)
        test_both(["z0z3","match_prob","gaussian"],lambda data: z_match_prob(gaussian(data)))

        def z_allmatch_prob(data):
            z_pre, z_suc, z_pre_aae = encode_and_split(data)
            z_match   = 1-np.abs(z_pre_aae-z_pre)
            return np.mean(np.prod(z_match,axis=1))

        test_both(["z0z3","allmatch_prob","vanilla"], z_allmatch_prob)
        test_both(["z0z3","allmatch_prob","gaussian"],lambda data: z_allmatch_prob(gaussian(data)))

        def z_xor(data):
            z     = self.encode(data)
            pre = z[:,0,...]
            suc = z[:,1,...]
            pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**opts)

            pos_precondition = self.regress([np.zeros((true_num_actions, *self.edim())),all_labels], **opts)
            neg_precondition = 1-self.regress([np.ones((true_num_actions, *self.edim())),all_labels], **opts)
            xor_precondition = pos_precondition * neg_precondition # 1 when both pos and neg are 1
            # now this is [A, N] matrix.
            return np.mean(np.sum(xor_precondition,axis=1))

        test_both(["z0z3","xor"], z_xor)

        def compiled_action(data):
            z     = self.encode(data)
            pre = z[:,0,...]
            suc = z[:,1,...]
            pre, suc, data, actions, histogram, true_num_actions, all_labels, action_ids = self._dump_actions_prologue(pre,suc,**opts)

            add_effect = self.apply([np.zeros((true_num_actions, *self.edim())),all_labels], **opts)
            del_effect = 1-self.apply([np.ones((true_num_actions, *self.edim())),all_labels], **opts)
            xor_effect = add_effect * del_effect # 1 when both add and del are 1
            pos_precondition = self.regress([np.zeros((true_num_actions, *self.edim())),all_labels], **opts)
            neg_precondition = 1-self.regress([np.ones((true_num_actions, *self.edim())),all_labels], **opts)
            xor_precondition = pos_precondition * neg_precondition # 1 when both pos and neg are 1
            # now this is [A, N] matrix.
            xor_bits = np.maximum(xor_effect, xor_precondition)
            xor_number_per_action = np.sum(xor_bits,axis=1)
            compiled_actions_per_action = 2 ** xor_number_per_action
            total_compiled_action = np.sum(compiled_actions_per_action)
            return total_compiled_action

        test_both(["z0z3","compiled_action"], compiled_action)

        return




class NormalizedLogitAddPreconditionMixin:
    def _build_primary(self,input_shape):
        self.pre_decoder_net = [
            MyFlatten(),
            Dense(np.prod(self.edim()),use_bias=False,kernel_regularizer=self.parameters["eff_regularizer"]),
            BN(),
        ]
        self.scaling_suc = BN()
        super()._build_primary(input_shape)

    def _regress(self,z_suc,action):
        l_eff     = Sequential(self.pre_decoder_net)(action)
        l_suc = self.scaling_suc(z_suc)
        l_pre_aae = add([l_suc,l_eff])
        z_pre_aae = self.build_bc()(l_pre_aae)
        return z_pre_aae



# effect + precondition variations

class NormalizedLogitAddBidirectionalMixin(NormalizedLogitAddPreconditionMixin, NormalizedLogitAddEffectMixin):
    pass


class UnidirectionalMixin(AddHocPreconditionMixin, EffectMixin):
    def plot_transitions(self,data,path,verbose=False,epoch=None):
        import os.path
        basename, ext = os.path.splitext(path)
        pre_path = basename+"_pre"+ext
        suc_path = basename+"_suc"+ext

        x = data
        z = self.encode(x)
        y = self.autoencode(x)

        x_pre, x_suc = x[:,0,...], x[:,1,...]
        z_pre, z_suc = z[:,0,...], z[:,1,...]
        y_pre, y_suc = y[:,0,...], y[:,1,...]
        action    = self.encode_action([z_pre,z_suc])
        z_suc_aae = self.apply([z_pre, action])
        y_suc_aae = self.decode(z_suc_aae)

        x_pre_r, x_suc_r = self.render(x_pre), self.render(x_suc)
        y_pre_r, y_suc_r = self.render(y_pre), self.render(y_suc)
        y_suc_aae_r = self.render(y_suc_aae)

        def diff(src,dst):
            return (dst - src + 1)/2


        self._plot(basename+"_transition_image"+ext,
                   [x_pre_r, x_suc_r,
                    y_pre_r, y_suc_r, y_suc_aae_r,],epoch=epoch)

        self._plot(basename+"_transition_image_diff"+ext,
                   [diff(x_pre_r,y_pre_r),
                    diff(x_suc_r,y_suc_r),
                    diff(x_suc_r,y_suc_aae_r),],epoch=epoch)

        self._plot(basename+"_transition_action"+ext,
                   [action,],epoch=epoch)

        self._plot(basename+"_transition_latent"+ext,
                   map(squarify,
                       [z_pre,     z_suc, z_suc_aae,]),epoch=epoch)

        self._plot(basename+"_transition_latent_diff"+ext,
                   map(squarify,
                       [diff(z_suc, z_suc_aae),]),epoch=epoch)

        self._plot(basename+"_transition_latent_eff"+ext,
                   map(squarify,
                       [diff(z_pre, z_suc),
                        diff(z_pre, z_suc_aae),]),epoch=epoch)

        p_b = self._encode_prob.predict(x)
        p_b_aae = self._encode_prob_aae.predict(x)
        p_b_pre, p_b_suc = p_b[:,0,...], p_b[:,1,...]
        _,       p_b_suc_aae = p_b_aae[:,0,...], p_b_aae[:,1,...]
        self._plot(basename+"_transition_prob"+ext,
                   map(squarify,
                       [p_b_pre,
                        p_b_suc,
                        p_b_suc_aae,]),epoch=epoch)
        self._plot(basename+"_transition_prob_diff"+ext,
                   map(squarify,
                       [diff(p_b_suc,p_b_suc_aae),]),epoch=epoch)
        return


class BidirectionalMixin(PreconditionMixin, EffectMixin):
    def plot_transitions(self,data,path,verbose=False,epoch=None):
        import os.path
        basename, ext = os.path.splitext(path)
        pre_path = basename+"_pre"+ext
        suc_path = basename+"_suc"+ext

        x = data
        z = self.encode(x)
        y = self.autoencode(x)

        x_pre, x_suc = x[:,0,...], x[:,1,...]
        z_pre, z_suc = z[:,0,...], z[:,1,...]
        y_pre, y_suc = y[:,0,...], y[:,1,...]
        action    = self.encode_action([z_pre,z_suc])
        z_pre_aae = self.regress([z_suc, action])
        y_pre_aae = self.decode(z_pre_aae)
        z_suc_aae = self.apply([z_pre, action])
        y_suc_aae = self.decode(z_suc_aae)

        def diff(src,dst):
            return (dst - src + 1)/2

        x_pre_r,     x_suc_r     = self.render(x_pre),     self.render(x_suc)
        y_pre_r,     y_suc_r     = self.render(y_pre),     self.render(y_suc)
        y_pre_aae_r, y_suc_aae_r = self.render(y_pre_aae), self.render(y_suc_aae)

        self._plot(basename+"_transition_image"+ext,
                   [x_pre_r, x_suc_r,
                    y_pre_r, y_suc_r,
                    y_pre_aae_r, y_suc_aae_r,],epoch=epoch)

        self._plot(basename+"_transition_image_diff"+ext,
                   [diff(x_pre_r,y_pre_r),
                    diff(x_suc_r,y_suc_r),
                    diff(x_pre_r,y_pre_aae_r),
                    diff(x_suc_r,y_suc_aae_r),],epoch=epoch)

        self._plot(basename+"_transition_action"+ext,
                   [action,],epoch=epoch)

        self._plot(basename+"_transition_latent"+ext,
                   map(squarify,
                       [z_pre,     z_suc,
                        z_pre_aae, z_suc_aae,]),epoch=epoch)

        self._plot(basename+"_transition_latent_diff"+ext,
                   map(squarify,
                       [diff(z_pre, z_pre_aae),
                        diff(z_suc, z_suc_aae),]),epoch=epoch)

        self._plot(basename+"_transition_latent_eff"+ext,
                   map(squarify,
                       [diff(z_pre, z_suc),
                        diff(z_pre, z_suc_aae),
                        diff(z_pre_aae, z_suc),]),epoch=epoch)

        p_b = self._encode_prob.predict(x)
        p_b_aae = self._encode_prob_aae.predict(x)
        p_b_pre, p_b_suc = p_b[:,0,...], p_b[:,1,...]
        p_b_pre_aae, p_b_suc_aae = p_b_aae[:,0,...], p_b_aae[:,1,...]
        self._plot(basename+"_transition_prob"+ext,
                   map(squarify,
                       [p_b_pre,
                        p_b_suc,
                        p_b_pre_aae,
                        p_b_suc_aae,]),epoch=epoch)
        self._plot(basename+"_transition_prob_diff"+ext,
                   map(squarify,
                       [diff(p_b_pre,p_b_pre_aae),
                        diff(p_b_suc,p_b_suc_aae),]),epoch=epoch)
        return



# AMA3 Space AE : Transition AE + Action AE double wielding! ###################

class BaseActionMixinAMA3(UnidirectionalMixin, BaseActionMixin):
    def _build_primary(self,input_shape):
        super()._build_primary(input_shape)

        x = self.net.input      # keras has a bug, we can"t make a new Input here
        _, x_pre, x_suc = dapply(x)
        z, z_pre, z_suc = dapply(self.encoder.output)
        y, y_pre, y_suc = dapply(self.autoencoder.output)

        action    = self._action(z_pre,z_suc)

        # set up direct loss (there are no other place it could be called)
        self.direct_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["direct_delay"]):self.parameters["direct"],}, name="direct")
        self.callbacks.append(LambdaCallback(on_epoch_end=self.direct_alpha.update))
        z_suc_aae = self._apply(z_pre,action)

        z_suc_aae = self.apply_direct_loss(z_suc,z_suc_aae)

        y_suc_aae = self._decode(z_suc_aae)

        state_loss_fn = self.output.loss

        rec_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["rec_delay"]):1,}, name="rec")
        aae_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["aae_delay"]):1,}, name="aae")
        self.callbacks.append(LambdaCallback(on_epoch_end=rec_alpha.update))
        self.callbacks.append(LambdaCallback(on_epoch_end=aae_alpha.update))

        def loss(dummy_x,dummy_y):
            return \
                rec_alpha.variable * state_loss_fn(x_pre, y_pre) + \
                rec_alpha.variable * state_loss_fn(x_suc, y_suc) + \
                aae_alpha.variable * state_loss_fn(x_suc, y_suc_aae)

        self.loss = loss
        self.net = Model(x, dmerge(y_pre, y_suc_aae))
        # note: adding the successor loss to self.net by add_loss and not defining a new net does not work
        # because Model.save only saves the weights that are included in the computation graph between input and output.
        self.encoder     = Model(x, z) # note : directly through the encoder, not AAE
        self.autoencoder = Model(x, y) # note : directly through the decoder, not AAE

        return

    pass


# AMA4  Space AE : Bidirectional model #########################################

class BaseActionMixinAMA4(BidirectionalMixin, BaseActionMixin):
    def _build_primary(self,input_shape):
        super()._build_primary(input_shape)

        x = self.net.input      # keras has a bug, we can"t make a new Input here
        _, x_pre, x_suc = dapply(x)
        z, z_pre, z_suc = dapply(self.encoder.output)
        y, y_pre, y_suc = dapply(self.autoencoder.output)

        action    = self._action(z_pre,z_suc)

        # set up direct loss (there are no other place it could be called)
        self.direct_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["direct_delay"]):self.parameters["direct"],}, name="direct")
        self.callbacks.append(LambdaCallback(on_epoch_end=self.direct_alpha.update))
        z_pre_aae = self._regress(z_suc,action)
        z_suc_aae = self._apply  (z_pre,action)

        z_pre_aae = self.apply_direct_loss(z_pre,z_pre_aae)
        z_suc_aae = self.apply_direct_loss(z_suc,z_suc_aae)

        y_pre_aae = self._decode(z_pre_aae)
        y_suc_aae = self._decode(z_suc_aae)

        state_loss_fn = self.output.loss

        rec_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["rec_delay"]):1,}, name="rec")
        aae_alpha = StepSchedule(schedule={0:0, (self.parameters["epoch"]*self.parameters["aae_delay"]):1,}, name="aae")
        self.callbacks.append(LambdaCallback(on_epoch_end=rec_alpha.update))
        self.callbacks.append(LambdaCallback(on_epoch_end=aae_alpha.update))

        def loss(dummy_x,dummy_y):
            return \
                rec_alpha.variable * state_loss_fn(x_pre, y_pre) + \
                rec_alpha.variable * state_loss_fn(x_suc, y_suc) + \
                aae_alpha.variable * state_loss_fn(x_pre, y_pre_aae) + \
                aae_alpha.variable * state_loss_fn(x_suc, y_suc_aae)

        self.loss = loss
        self.net = Model(x, dmerge(y_pre_aae, y_suc_aae))
        # note: adding the successor loss to self.net by add_loss and not defining a new net does not work
        # because Model.save only saves the weights that are included in the computation graph between input and output.
        self.encoder     = Model(x, z) # note : directly through the encoder, not AAE
        self.autoencoder = Model(x, y) # note : directly through the decoder, not AAE

        return

    pass


# AMA3+ Space AE : Space AE with correct ELBO ##################################

class BaseActionMixinAMA3Plus(UnidirectionalMixin, BaseActionMixin):
    def _save(self,path=""):
        # saved separately so that we can switch between loading or not loading it.
        # since the weights are fixed size, loading it with a different input shape causes an error.
        super()._save(path)
        print("saving additional networks")
        import os.path
        np.savez_compressed(self.local(os.path.join(path,f"p_a_z0_net.npz")),*self.p_a_z0_net[0].get_weights())

    def _load(self,path=""):
        # loaded separately so that we can switch between loading or not loading it.
        # since the weights are fixed size, loading it with a different input shape causes an error.
        # reload_with_shape does not call _load.
        super()._load(path)
        print("loading additional networks")
        import os.path
        try:
            with np.load(self.local(os.path.join(path,f"p_a_z0_net.npz"))) as data:
                self.p_a_z0_net[0].set_weights([ data[key] for key in data.files ])
        except FileNotFoundError:
            print("failed to find weights for additional networks")

    def _build_around(self,input_shape):
        A  = self.parameters["A"]
        self.p_a_z0_net = [
            Dense(A),
            Reshape(self.adim()),
        ]
        def update_dynamics_training_flag(epoch, logs=None):
            if epoch == self.parameters["kl_cycle_start"]:
                print(f"epoch {epoch}: freezing the decoder")
                for layer in self.decoder_net:
                    layer.trainable = False
                # force compilation
                self._compile(self.optimizers)
        self.callbacks.append(LambdaCallback(on_epoch_begin = update_dynamics_training_flag))
        super()._build_around(input_shape)

    def _build_primary(self,input_shape):

        x = Input(shape=(2,*input_shape))
        _, x_pre, x_suc = dapply(x)
        z, z_pre, z_suc = dapply(x, self._encode)
        y, y_pre, y_suc = dapply(z, self._decode)

        # to generate a correct ELBO, input to action should be deterministic
        (l_pre,     ), _ = z_pre.variational_source # see Variational class
        (l_suc,     ), _ = z_suc.variational_source # see Variational class
        p_pre = wrap(l_pre, K.sigmoid(l_pre))
        p_suc = wrap(l_suc, K.sigmoid(l_suc))
        # note: _action takes a probability, but feeding 0/1 data in test time is fine (0/1 can be seen as probabilities)
        action    = self._action(p_pre,p_suc)
        z_suc_aae = self._apply(z_pre,action)
        y_suc_aae = self._decode(z_suc_aae)
        z_aae = dmerge(z_pre, z_suc_aae)
        y_aae = dmerge(y_pre, y_suc_aae)

        (l_action,  ), _ = action.variational_source # see Variational class
        (l_suc_aae, ), _ = z_suc_aae.variational_source # see Variational class

        p         = dmerge(p_pre, p_suc)
        p_suc_aae = wrap(l_suc_aae, K.sigmoid(l_suc_aae))
        p_aae     = dmerge(p_pre, p_suc_aae)

        pdiff_z1z2 = K.mean(K.abs(p_suc - p_suc_aae),axis=-1)
        pdiff_z0z1 = K.mean(K.abs(p_pre - p_suc),axis=-1)
        pdiff_z0z2 = K.mean(K.abs(p_pre - p_suc_aae),axis=-1)

        kl_z0 = z_pre.loss(l_pre, p=self.parameters["zerosuppress"])
        kl_a_z0 = action.loss(l_action, logit_p=Sequential(self.p_a_z0_net)(z_pre))
        kl_z1z2 = z_suc_aae.loss(l_suc, logit_p=l_suc_aae)
        _rec = self.output.loss
        x0y0 = _rec(x_pre,y_pre)
        x1y1 = _rec(x_suc,y_suc)
        x1y2 = _rec(x_suc,y_suc_aae)
        ama3_forward_loss1  = self.parameters["beta_z"] * kl_z0 + x0y0 + kl_a_z0 + self.parameters["beta_d"] * kl_z1z2 + x1y1
        ama3_forward_loss2  = self.parameters["beta_z"] * kl_z0 + x0y0 + kl_a_z0 + x1y2
        total_loss = (ama3_forward_loss1 + ama3_forward_loss2)/2
        ama3_forward_elbo1  = kl_z0 + x0y0 + kl_a_z0 + kl_z1z2 + x1y1
        ama3_forward_elbo2  = kl_z0 + x0y0 + kl_a_z0 + x1y2
        elbo = (ama3_forward_elbo1 + ama3_forward_elbo2)/2
        self.add_metric("pdiff_z1z2",pdiff_z1z2)
        self.add_metric("pdiff_z0z1",pdiff_z0z1)
        self.add_metric("pdiff_z0z2",pdiff_z0z2)
        self.add_metric("kl_z0",kl_z0)
        self.add_metric("kl_a_z0",kl_a_z0)
        self.add_metric("kl_z1z2",kl_z1z2)
        self.add_metric("x0y0",x0y0)
        self.add_metric("x1y1",x1y1)
        self.add_metric("x1y2",x1y2)
        self.add_metric("elbo",elbo)
        def loss(*args):
            return total_loss
        self.loss = loss
        # note: original z does not work because Model.save only saves the weights that are included in the computation graph between input and output.
        self.net = Model(x, y_aae)
        self.encoder     = Model(x, z) # note : directly through the encoder, not AAE
        self.autoencoder = Model(x, y) # note : directly through the decoder, not AAE

        # verify the note above : self.autoencoder.weights does not contain weights for AAE
        # print(self.net.weights)
        # print(self.autoencoder.weights)

        # for plotting
        self._encode_prob     = Model(x, p) # note : directly through the encoder, not AAE
        self._encode_prob_aae = Model(x, p_aae) # note : directly through the encoder, not AAE

        return

    def evaluate(self,*args,**kwargs):
        metrics = { k:v for k,v in zip(self.net.metrics_names,
                                       ensure_list(self.net.evaluate(*args,**kwargs)))}
        return [metrics["elbo"],metrics["loss"],metrics["x0y0"]+metrics["x1y1"]/2+metrics["x1y2"]/2, metrics["kl_z1z2"], metrics["kl_z0"]]

    pass



# AMA4+ Space AE : Bidirectional model with correct ELBO #######################

class BaseActionMixinAMA4Plus(BidirectionalMixin, BaseActionMixin):
    def _save(self,path=""):
        # saved separately so that we can switch between loading or not loading it.
        # since the weights are fixed size, loading it with a different input shape causes an error.
        super()._save(path)
        print("saving additional networks")
        import os.path
        np.savez_compressed(self.local(os.path.join(path,f"p_a_z0_net.npz")),*self.p_a_z0_net[0].get_weights())
        np.savez_compressed(self.local(os.path.join(path,f"p_a_z1_net.npz")),*self.p_a_z1_net[0].get_weights())

    def _load(self,path=""):
        # loaded separately so that we can switch between loading or not loading it.
        # since the weights are fixed size, loading it with a different input shape causes an error.
        # reload_with_shape does not call _load.
        super()._load(path)
        print("loading additional networks")
        import os.path
        try:
            with np.load(self.local(os.path.join(path,f"p_a_z0_net.npz"))) as data:
                self.p_a_z0_net[0].set_weights([ data[key] for key in data.files ])
            with np.load(self.local(os.path.join(path,f"p_a_z1_net.npz"))) as data:
                self.p_a_z1_net[0].set_weights([ data[key] for key in data.files ])
        except FileNotFoundError:
            print("failed to find weights for additional networks")

    def _build_around(self,input_shape):
        A  = self.parameters["A"]
        self.p_a_z0_net = [
            Dense(A),
            Reshape(self.adim()),
        ]
        self.p_a_z1_net = [
            Dense(A),
            Reshape(self.adim()),
        ]
        initial_weights = []
        def save_initial_weight(epoch, logs=None):
            for layer in self.encoder_net:
                initial_weights.append(layer.get_weights())
        def reset_initial_weight(epoch, logs=None):
            if epoch == self.parameters["kl_cycle_start"]:
                for layer,weights in zip(self.encoder_net,initial_weights):
                    layer.set_weights(weights)
        # self.callbacks.append(LambdaCallback(on_train_begin = save_initial_weight))
        # self.callbacks.append(LambdaCallback(on_epoch_begin = reset_initial_weight))
        def update_dynamics_training_flag(epoch, logs=None):
            if epoch == self.parameters["kl_cycle_start"]:
                print(f"epoch {epoch}: freezing the decoder")
                for layer in (*self.decoder_net,
                              *self.eff_decoder_net,
                              *self.pre_decoder_net):
                    layer.trainable = False
                # force compilation
                self._compile(self.optimizers)
        # self.callbacks.append(LambdaCallback(on_epoch_begin = update_dynamics_training_flag))
        super()._build_around(input_shape)

    def _build_primary(self,input_shape):

        x = Input(shape=(2,*input_shape))
        _, x_pre, x_suc = dapply(x)
        z, z_pre, z_suc = dapply(x, self._encode)
        y, y_pre, y_suc = dapply(z, self._decode)
        # to generate a correct ELBO, input to action should be deterministic
        (l_pre,     ), _ = z_pre.variational_source # see Variational class
        (l_suc,     ), _ = z_suc.variational_source # see Variational class
        p_pre = wrap(l_pre, K.sigmoid(l_pre))
        p_suc = wrap(l_suc, K.sigmoid(l_suc))
        # note: _action takes a probability, but feeding 0/1 data in test time is fine (0/1 can be seen as probabilities)
        action    = self._action (p_pre,p_suc)
        z_suc_aae = self._apply  (z_pre,action)
        z_pre_aae = self._regress(z_suc,action)
        y_suc_aae = self._decode(z_suc_aae)
        y_pre_aae = self._decode(z_pre_aae)
        z_aae = dmerge(z_pre_aae, z_suc_aae)
        y_aae = dmerge(y_pre_aae, y_suc_aae)

        (l_action,  ), _ = action.variational_source # see Variational class
        (l_suc_aae, ), _ = z_suc_aae.variational_source # see Variational class
        (l_pre_aae, ), _ = z_pre_aae.variational_source # see Variational class

        p         = dmerge(p_pre, p_suc)
        p_pre_aae = wrap(l_pre_aae, K.sigmoid(l_pre_aae))
        p_suc_aae = wrap(l_suc_aae, K.sigmoid(l_suc_aae))
        p_aae     = dmerge(p_pre_aae, p_suc_aae)

        pdiff_z1z2 = K.mean(K.abs(p_suc - p_suc_aae),axis=-1)
        pdiff_z0z3 = K.mean(K.abs(p_pre - p_pre_aae),axis=-1)
        pdiff_z0z1 = K.mean(K.abs(p_pre - p_suc),axis=-1)
        pdiff_z0z2 = K.mean(K.abs(p_pre - p_suc_aae),axis=-1)

        kl_z0 = z_pre.loss(l_pre, p=self.parameters["zerosuppress"])
        kl_z1 = z_suc.loss(l_suc, p=self.parameters["zerosuppress"])
        kl_a_z0 = action.loss(l_action, logit_p=Sequential(self.p_a_z0_net)(z_pre))
        kl_a_z1 = action.loss(l_action, logit_p=Sequential(self.p_a_z1_net)(z_suc))
        kl_z1z2 = z_pre_aae.loss(l_pre, logit_p=l_pre_aae)
        kl_z0z3 = z_suc_aae.loss(l_suc, logit_p=l_suc_aae)
        _rec = self.output.loss
        x0y0 = _rec(x_pre,y_pre)
        x1y1 = _rec(x_suc,y_suc)
        x0y3 = _rec(x_pre,y_pre_aae)
        x1y2 = _rec(x_suc,y_suc_aae)
        ama3_forward_loss1  = self.parameters["beta_z"] * kl_z0 + x0y0 + kl_a_z0 + self.parameters["beta_d"] * kl_z1z2 + x1y1
        ama3_forward_loss2  = self.parameters["beta_z"] * kl_z0 + x0y0 + kl_a_z0 + x1y2
        ama3_backward_loss1 = self.parameters["beta_z"] * kl_z1 + x1y1 + kl_a_z1 + self.parameters["beta_d"] * kl_z0z3 + x0y0
        ama3_backward_loss2 = self.parameters["beta_z"] * kl_z1 + x1y1 + kl_a_z1 + x0y3
        total_loss = (ama3_forward_loss1 + ama3_forward_loss2 + ama3_backward_loss1 + ama3_backward_loss2)/4 
        ama3_forward_elbo1  = kl_z0 + x0y0 + kl_a_z0 + kl_z1z2 + x1y1
        ama3_forward_elbo2  = kl_z0 + x0y0 + kl_a_z0 + x1y2
        ama3_backward_elbo1 = kl_z1 + x1y1 + kl_a_z1 + kl_z0z3 + x0y0
        ama3_backward_elbo2 = kl_z1 + x1y1 + kl_a_z1 + x0y3
        elbo = (ama3_forward_elbo1 + ama3_forward_elbo2 + ama3_backward_elbo1 + ama3_backward_elbo2)/4
        self.add_metric("pdiff_z1z2",pdiff_z1z2)
        self.add_metric("pdiff_z0z3",pdiff_z0z3)
        self.add_metric("pdiff_z0z1",pdiff_z0z1)
        self.add_metric("pdiff_z0z2",pdiff_z0z2)
        self.add_metric("kl_z0",kl_z0)
        self.add_metric("kl_z1",kl_z1)
        self.add_metric("kl_a_z0",kl_a_z0)
        self.add_metric("kl_a_z1",kl_a_z1)
        self.add_metric("kl_z1z2",kl_z1z2)
        self.add_metric("kl_z0z3",kl_z0z3)
        self.add_metric("x0y0",x0y0)
        self.add_metric("x1y1",x1y1)
        self.add_metric("x0y3",x0y3)
        self.add_metric("x1y2",x1y2)
        self.add_metric("elbo",elbo)
        def loss(*args):
            return total_loss
        self.loss = loss

        # note: original z does not work because Model.save only saves the weights that are included in the computation graph between input and output.
        self.net = Model(x, y_aae)
        self.encoder     = Model(x, z) # note : directly through the encoder, not AAE
        self.autoencoder = Model(x, y) # note : directly through the decoder, not AAE

        # verify the note above : self.autoencoder.weights does not contain weights for AAE
        # print(self.net.weights)
        # print(self.autoencoder.weights)

        # for plotting
        self._encode_prob     = Model(x, p) # note : directly through the encoder, not AAE
        self._encode_prob_aae = Model(x, p_aae) # note : directly through the encoder, not AAE

        return

    def evaluate(self,*args,**kwargs):
        metrics = { k:v for k,v in zip(self.net.metrics_names,
                                       ensure_list(self.net.evaluate(*args,**kwargs)))}
        return [metrics["elbo"],metrics["loss"],metrics["x0y0"]+metrics["x1y1"]/2+metrics["x1y2"]/2, metrics["kl_z1z2"], metrics["kl_z0"]]

    pass





################################################################################
# Concrete Class Instantiations

# Zero-sup SAE #################################################################

# ICAPS19 paper
class ZeroSuppressStateAE              (ZeroSuppressMixin, StateAE):
    pass

# Transition SAE ################################################################

# earlier attempts to "sparcify" the transitions. No longer used
class VanillaTransitionAE(              ZeroSuppressMixin, TransitionWrapper, StateAE):
    pass

class HammingTransitionAE(HammingMixin, ZeroSuppressMixin, TransitionWrapper, StateAE):
    pass
class CosineTransitionAE (CosineMixin,  ZeroSuppressMixin, TransitionWrapper, StateAE):
    pass
class PoissonTransitionAE(PoissonMixin, ZeroSuppressMixin, TransitionWrapper, StateAE):
    pass


# IJCAI2020 paper : AMA3
# note: ZeroSuppressMixin must come before TransitionWrapper
# because the loss must be set up after the correct self.encoder is set by _build in TransitionWrapper
class ConcreteDetConditionalEffectTransitionAE              (ZeroSuppressMixin, DetActionMixin, ConditionalEffectMixin,        BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Vanilla Space AE"""
    pass
class ConcreteDetBoolMinMaxEffectTransitionAE               (ZeroSuppressMixin, DetActionMixin, BoolMinMaxEffectMixin,         BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Cube-Space AE with naive discrete effects (not BTL)"""
    pass
class ConcreteDetBoolSmoothMinMaxEffectTransitionAE         (ZeroSuppressMixin, DetActionMixin, BoolSmoothMinMaxEffectMixin,   BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Cube-Space AE with naive discrete effects with smooth min/max"""
    pass
class ConcreteDetLogitAddEffectTransitionAE                 (ZeroSuppressMixin, DetActionMixin, LogitAddEffectMixin,           BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Cube-Space AE without BatchNorm"""
    pass
class ConcreteDetLogitAddEffect2TransitionAE                (ZeroSuppressMixin, DetActionMixin, LogitAddEffect2Mixin,          BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Cube-Space AE without BatchNorm for the current state but with BatchNorm for effects"""
    pass
class ConcreteDetNormalizedLogitAddEffectTransitionAE       (ZeroSuppressMixin, DetActionMixin, NormalizedLogitAddEffectMixin, BaseActionMixinAMA3, TransitionWrapper, StateAE):
    """Final Cube-Space AE implementation in IJCAI2020"""
    pass


CubeSpaceAE_AMA3 = ConcreteDetNormalizedLogitAddEffectTransitionAE


# AMA4 : bidirectional extention of AMA3. It does not have a correct elbo, but it trains reasonably
class ConcreteDetNormalizedLogitAddBidirectionalTransitionAE(ZeroSuppressMixin, DetActionMixin, NormalizedLogitAddBidirectionalMixin, BaseActionMixinAMA4, TransitionWrapper, StateAE):
    """Bidirectional Cube-Space AE implementation; not appearing in any literature"""
    pass


CubeSpaceAE_AMA4 = ConcreteDetNormalizedLogitAddBidirectionalTransitionAE

# note: ConvolutionLiftedBidirectionalMixin must come first in order to set self.parameters["N"] in _build_around

# AMA3 plus : Correction of AMA3 that optimizes the network using correct ELBO.
class ConcreteDetLogitAddEffectTransitionAEPlus(DetActionMixin, LogitAddEffectMixin, BaseActionMixinAMA3Plus, TransitionWrapper, StateAE):
    """Cube-Space AE implementation in JAIR"""
    pass

class ConcreteDetLogitAddEffect2TransitionAEPlus(DetActionMixin, LogitAddEffect2Mixin, BaseActionMixinAMA3Plus, TransitionWrapper, StateAE):
    """Cube-Space AE implementation in JAIR"""
    pass

class ConcreteDetNormalizedLogitAddEffectTransitionAEPlus(DetActionMixin, NormalizedLogitAddEffectMixin, BaseActionMixinAMA3Plus, TransitionWrapper, StateAE):
    """Cube-Space AE implementation in JAIR"""
    pass

class ConvolutionalConcreteDetNormalizedLogitAddEffectTransitionAEPlus(StridedConvolutionalMixin, ConcreteDetNormalizedLogitAddEffectTransitionAEPlus):
    pass

CubeSpaceAE_AMA3Plus = ConcreteDetNormalizedLogitAddEffectTransitionAEPlus
CubeSpaceAE_AMA3Conv = ConvolutionalConcreteDetNormalizedLogitAddEffectTransitionAEPlus


# AMA4 plus : Correction of AMA4 that optimizes the network using correct ELBO.
class ConcreteDetNormalizedLogitAddBidirectionalTransitionAEPlus(DetActionMixin, NormalizedLogitAddBidirectionalMixin, BaseActionMixinAMA4Plus, TransitionWrapper, StateAE):
    """Bidirectional Cube-Space AE implementation in JAIR"""
    pass

class ConvolutionalConcreteDetNormalizedLogitAddBidirectionalTransitionAEPlus(StridedConvolutionalMixin,ConcreteDetNormalizedLogitAddBidirectionalTransitionAEPlus):
    pass

CubeSpaceAE_AMA4Plus = ConcreteDetNormalizedLogitAddBidirectionalTransitionAEPlus
CubeSpaceAE_AMA4Conv = ConvolutionalConcreteDetNormalizedLogitAddBidirectionalTransitionAEPlus



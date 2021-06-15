import keras.initializers
import keras.backend.tensorflow_backend as K
from keras.layers import *
from keras.initializers import Initializer
import numpy as np
import tensorflow as tf
debug = False
# debug = True

def Print(msg=None):
    def printer(x):
        if msg:
            print(x,":",msg)
        else:
            print(x)
        return x
    return Lambda(printer)

def list_layer_io(net):
    from keras.models import Model
    print(net)
    if isinstance(net, list):
        for subnet in net:
            list_layer_io(subnet)
    elif isinstance(net, Model):
        net.summary()
    elif isinstance(net, Layer):
        print("  <-")
        for i in range(len(net._inbound_nodes)):
            print(net._get_node_attribute_at_index(i, 'input_tensors', 'input'))
        print("  ->")
        for i in range(len(net._inbound_nodes)):
            print(net._get_node_attribute_at_index(i, 'output_tensors', 'output'))
        # print(net.input)
    else:
        print("nothing can be displayed")


debug_level = 0
def Sequential (array):
    from functools import reduce
    def apply1(arg,f):
        global debug_level
        if debug:
            print(" "*debug_level+"applying {}({})".format(f,arg))
        debug_level += 2
        try:
            result = f(arg)
        finally:
            debug_level -= 2
        if debug:
            try:
                print(" "*debug_level,"->",K.int_shape(result), K.shape(result))
            except:
                print(" "*debug_level,"->",result)
        return result
    return lambda x: reduce(apply1, array, x)

def ConditionalSequential (array, condition, **kwargs):
    from functools import reduce
    def apply1(arg,f):
        if debug:
            print("applying {}({})".format(f,arg))
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

def wrap(x,y,**kwargs):
    "wrap arbitrary operation"
    return Lambda(lambda x:y,**kwargs)(x)


def Densify(layers):
    "Apply layers in a densenet-like manner."
    def densify_fn(x):
        def rec(x,layers):
            if len(layers) == 0:
                return x
            else:
                layer, *rest = layers
                result = layer(x)
                return rec(concatenate([x,result],axis=-1), rest)
        return rec(x,layers)
    return densify_fn


class MyFlatten(Layer):
    """MyFlattens the input. Does not affect the batch size.

    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            The purpose of this argument is to preserve weight
            ordering when switching a model from one data format
            to another.
            `channels_last` corresponds to inputs with shape
            `(batch, ..., channels)` while `channels_first` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".

    # Example

    ```python
        model = Sequential()
        model.add(Conv2D(64, (3, 3),
                         input_shape=(3, 32, 32), padding='same',))
        # now: model.output_shape == (None, 64, 32, 32)

        model.add(MyFlatten())
        # now: model.output_shape == (None, 65536)
    ```
    """

    def __init__(self, data_format=None, **kwargs):
        super(MyFlatten, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.data_format = K.normalize_data_format(data_format)

    def compute_output_shape(self, input_shape):
        if not all(input_shape[1:]):
            raise ValueError('The shape of the input to "MyFlatten" '
                             'is not fully defined '
                             '(got ' + str(input_shape[1:]) + '). '
                             'Make sure to pass a complete "input_shape" '
                             'or "batch_input_shape" argument to the first '
                             'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))

    def call(self, inputs):
        if self.data_format == 'channels_first':
            # Ensure works for any dim
            permutation = [0]
            permutation.extend([i for i in
                                range(2, K.ndim(inputs))])
            permutation.append(1)
            inputs = K.permute_dimensions(inputs, permutation)

        return K.batch_flatten(inputs)

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(MyFlatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def flatten1D(x):
    def fn(x):
        if K.ndim(x) == 3:
            return x
        elif K.ndim(x) > 3:
            s = K.shape(x)
            return K.reshape(x, [K.shape(x)[0],int(np.prod(K.int_shape(x)[1:-1])),K.int_shape(x)[-1]])
        else:
            raise Exception(f"unsupported shape {K.shape(x)}")
    return Lambda(fn)(x)
def flatten2D(x):
    def fn(x):
        if K.ndim(x) == 4:
            return x
        elif K.ndim(x) > 4:
            return K.reshape(x, [K.shape(x)[0],int(np.prod(K.int_shape(x)[1:-2])),K.int_shape(x)[-2],K.int_shape(x)[-1]])
        else:
            raise Exception(f"unsupported shape {K.shape(x)}")
    return Lambda(fn)(x)


def set_trainable (model, flag):
    from collections.abc import Iterable
    if isinstance(model, Iterable):
        for l in model:
            set_trainable(l, flag)
    elif hasattr(model, "layers"):
        set_trainable(model.layers,flag)
    else:
        model.trainable = flag

def sort_binary(x):
    x = x.round().astype(np.uint64)
    steps = np.arange(start=x.shape[-1]-1, stop=-1, step=-1, dtype=np.uint64)
    two_exp = (2 << steps)//2
    x_int = np.sort(np.dot(x, two_exp))
    # print(x_int)
    xs=[]
    for i in range(((x.shape[-1]-1)//8)+1):
        xs.append(x_int % (2**8))
        x_int = x_int // (2**8)
    xs.reverse()
    # print(xs)
    tmp = np.stack(xs,axis=-1)
    # print(tmp)
    tmp = np.unpackbits(tmp.astype(np.uint8),-1)
    # print(tmp)
    return tmp[...,-x.shape[-1]:]

# tests
# sort_binary(np.array([[[1,0,0,0],[0,1,0,0],],[[0,1,0,0],[1,0,0,0]]]))
# sort_binary(np.array([[[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],],
#                       [[0,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0]]]))

def count_params(model):
    from keras.utils.layer_utils import count_params
    model._check_trainable_weights_consistency()
    if hasattr(model, '_collected_trainable_weights'):
        trainable_count = count_params(model._collected_trainable_weights)
    else:
        trainable_count = count_params(model.trainable_weights)
    return trainable_count

from keras.callbacks import Callback

class HistoryBasedEarlyStopping(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f'\nEpoch {self.stopped_epoch}: early stopping {type(self)}')
            print('history:',self.history)

class GradientEarlyStopping(HistoryBasedEarlyStopping):
    def __init__(self, monitor='val_loss',
                 min_grad=-0.0001, sample_epochs=20, verbose=0, smooth=3):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.min_grad = min_grad
        self.history = []
        self.sample_epochs = sample_epochs
        self.stopped_epoch = 0
        assert sample_epochs >= 2
        if sample_epochs > smooth*2:
            self.smooth = smooth
        else:
            print("sample_epochs is too small for smoothing!")
            self.smooth = sample_epochs//2

    def gradient(self):
        h = np.array(self.history)
        
        # e.g. when smooth = 3, take the first/last 3 elements, average them over 3,
        # take the difference, then divide them by the epoch(== length of the history)
        return (h[-self.smooth:] - h[:self.smooth]).mean()/self.sample_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        import warnings
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        self.history.append(current) # to the last
        if len(self.history) > self.sample_epochs:
            self.history.pop(0) # from the front
            if self.gradient() >= self.min_grad:
                self.model.stop_training = True
                self.stopped_epoch = epoch

class ChangeEarlyStopping(HistoryBasedEarlyStopping):
    "Stops when the training gets stabilized: when the change of the past epochs are below a certain threshold"
    def __init__(self, monitor='val_loss',
                 threshold=0.00001, epoch_start=0, sample_epochs=20, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.threshold = threshold
        self.history = []
        self.epoch_start = epoch_start
        self.sample_epochs = sample_epochs
        self.stopped_epoch = 0

    def change(self):
        return (np.amax(self.history)-np.amin(self.history))

    def on_epoch_end(self, epoch, logs=None):
        import warnings
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        self.history.append(current) # to the last
        if len(self.history) > self.sample_epochs:
            self.history.pop(0) # from the front
            if (self.change() <= self.threshold) and (self.epoch_start <= epoch) :
                self.model.stop_training = True
                self.stopped_epoch = epoch

class LinearEarlyStopping(HistoryBasedEarlyStopping):
    "Stops when the value goes above the linearly decreasing upper bound"
    def __init__(self,
                 epoch_end,
                 epoch_start=0,
                 monitor='val_loss',
                 ub_ratio_start=1.0, ub_ratio_end=0.0, # note: relative to the loss at epoch 0
                 target_value=None,
                 sample_epochs=20, verbose=0):
        super().__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.history = []
        self.epoch_end     = epoch_end
        self.epoch_start   = epoch_start
        self.ub_ratio_end     = ub_ratio_end
        self.ub_ratio_start   = ub_ratio_start
        self.sample_epochs = sample_epochs
        self.stopped_epoch = 0
        self.value_start = float("inf")
        if target_value is not None:
            self.value_end = target_value
        else:
            self.value_end = 0.0

    def on_epoch_end(self, epoch, logs=None):
        import warnings
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if epoch == self.epoch_start:
            self.value_start = current

        progress_ratio = (epoch - self.epoch_start) / (self.epoch_end - self.epoch_start)
        ub_ratio = self.ub_ratio_start + (self.ub_ratio_end - self.ub_ratio_start) * progress_ratio
        ub = (self.value_start - self.value_end) * ub_ratio + self.value_end

        self.history.append(current) # to the last
        if len(self.history) > self.sample_epochs:
            self.history.pop(0) # from the front
            if (np.median(self.history) >= ub) and (self.epoch_start <= epoch) :
                self.model.stop_training = True
                self.stopped_epoch = epoch

class ExplosionEarlyStopping(HistoryBasedEarlyStopping):
    "Stops when the value goes above the upper bound, which is set to a very large value (1e8 by default)"
    def __init__(self,
                 epoch_end,
                 epoch_start=0,
                 monitor='val_loss',
                 sample_epochs=20, verbose=0):
        super().__init__()
        self.monitor       = monitor
        self.verbose       = verbose
        self.history       = []
        self.epoch_end     = epoch_end
        self.epoch_start   = epoch_start
        self.sample_epochs = sample_epochs
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        import warnings
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)
        if epoch == self.epoch_start:
            self.ub = current * 10
        if np.isnan(current) :
            self.model.stop_training = True
            self.stopped_epoch = epoch
            return
        self.history.append(current) # to the last
        if len(self.history) > self.sample_epochs:
            self.history.pop(0) # from the front
            if (np.median(self.history) >= self.ub) and (epoch >= self.epoch_start) :
                self.model.stop_training = True
                self.stopped_epoch = epoch

def anneal_rate(epoch,min=0.1,max=5.0):
    assert epoch > 0
    import math
    return math.log(max/min) / epoch

take_true_counter = 0
def take_true(name="take_true"):
    global take_true_counter
    take_true_counter += 1
    return Lambda(lambda x: x[:,:,0], name="{}_{}".format(name,take_true_counter))

# sign function with straight-through estimator
sign_counter = 0
def sign(name="sign"):
    global sign_counter
    sign_counter += 1
    import tensorflow as tf
    def fn(x):
        g = tf.get_default_graph()
        with g.gradient_override_map({"Sign": "Identity"}):
            return tf.sign(x)
    return Lambda(fn,name="{}_{}".format(name,sign_counter))

# heavyside step function with straight-through estimator
heavyside_counter = 0
def heavyside(name="heavyside"):
    global heavyside_counter
    heavyside_counter += 1
    import tensorflow as tf
    def fn(x):
        g = tf.get_default_graph()
        with g.gradient_override_map({"Sign": "Identity"}):
            return (tf.sign(x)+1)/2
    return Lambda(fn,name="{}_{}".format(name,heavyside_counter))

# argmax function with straight-through estimator
argmax_counter = 0
def argmax(name="argmax"):
    global argmax_counter
    argmax_counter += 1
    import tensorflow as tf
    def fn(x):
        g = tf.get_default_graph()
        with g.gradient_override_map({"Sign": "Identity"}):
            return (tf.sign(x-K.max(x,axis=-1,keepdims=True)+1e-20)+1)/2
    return Lambda(fn,name="{}_{}".format(name,argmax_counter))

# sigmoid that becomes a step function in the test time
rounded_sigmoid_counter = 0
def rounded_sigmoid(name="rounded_sigmoid"):
    global rounded_sigmoid_counter
    rounded_sigmoid_counter += 1
    return Lambda(lambda x: K.in_train_phase(K.sigmoid(x), K.round(K.sigmoid(x))),
                  name="{}_{}".format(name,rounded_sigmoid_counter))

# softmax that becomes an argmax function in the test time
rounded_softmax_counter = 0
def rounded_softmax(name="rounded_softmax"):
    global rounded_softmax_counter
    rounded_softmax_counter += 1
    return Lambda(lambda x: K.in_train_phase(K.softmax(x), K.one_hot(K.argmax( x ), K.int_shape(x)[-1])),
                  name="{}_{}".format(name,rounded_softmax_counter))

# is a maximum during the test time
def smooth_max(*args):
    return K.in_train_phase(K.logsumexp(K.stack(args,axis=0), axis=0)-K.log(2.0), K.maximum(*args))

# is a minimum during the test time
def smooth_min(*args):
    return K.in_train_phase(-K.logsumexp(-K.stack(args,axis=0), axis=0)+K.log(2.0), K.minimum(*args))

stclip_counter = 0
def stclip(min_value,high_value,name="stclip"):
    "clip with straight-through gradient"
    global stclip_counter
    stclip_counter += 1
    import tensorflow as tf
    def fn(x):
        x_clip = K.clip(x, min_value, high_value)
        return K.stop_gradient(x_clip - x) + x
    return Lambda(fn,name="{}_{}".format(name,stclip_counter))


def delay(self, x, amount):
    switch = K.variable(0)
    def fn(epoch,log):
        if epoch > amount:
            K.set_value(switch, 1)
        else:
            K.set_value(switch, 0)
    self.callbacks.append(LambdaCallback(on_epoch_end=fn))
    return switch * x

dmerge_counter = 0
def dmerge(x1, x2):
    """Take a pair of batched tensors (e.g. shape : [B,D,E,F]), then concatenate
    them as a batch of pairs (e.g. shape : [B,2,D,E,F])."""
    global dmerge_counter
    dmerge_counter += 1
    return concatenate([wrap(x1, x1[:,None,...]),wrap(x2, x2[:,None,...])],axis=1,name="dmerge_{}".format(dmerge_counter))

def dapply(x,fn=None):
    """Take a batch of pairs of elements (e.g. shape : [B,2,D,E,F]), apply fn to each half
([B,D,E,F]), then concatenate the result into pairs ([B,2,result_shape])."""
    x1 = wrap(x,x[:,0,...])
    x2 = wrap(x,x[:,1,...])
    if fn is not None:
        y1 = fn(x1)
        y2 = fn(x2)
    else:
        y1 = x1
        y2 = x2
    y = dmerge(y1, y2)
    return y, y1, y2


class Variational(Layer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    def __call__(self,*args,**kwargs):
        result = super().__call__(*args,**kwargs)
        result.variational_source = args, kwargs
        result.loss = self.loss
        # now you can compute a KL loss using the resultng tensor T like T.loss(...)
        return result

class Gaussian(Variational):
    """Gaussian variational layer.

Call this instance with mean_log_var which represents q.
Its first half in the last dimension represents the mean, and the second half the log-variance.
It adds the KL divergence loss against p=N(0,1).

Optionally, you can call the instance with two argumetns (mean_log_var_q, mean_log_var_p),
the latter of which represents a target / prior distribution p.
Then it adds a KL divergence loss KL(q=N(mu_q,sigma_q)||p=N(mu_p,sigma_p)).
It is useful when you want to match two distributions.
    """
    count = 0

    def sample(self, mean_log_var_q):
        sym_shape = K.shape(mean_log_var_q)
        shape = K.int_shape(mean_log_var_q)
        dims = [sym_shape[i] for i in range(len(shape)-1)]
        dim = shape[-1]//2
        mean_q    = mean_log_var_q[...,:dim]
        log_var_q = mean_log_var_q[...,dim:]
        noise = K.exp(0.5 * log_var_q) * K.random_normal(shape=(*dims, dim))
        return K.in_train_phase(mean_q + noise, mean_q)

    def loss(self, mean_log_var_q, mean_log_var_p=None):
        sym_shape = K.shape(mean_log_var_q)
        shape = K.int_shape(mean_log_var_q)
        dims = [sym_shape[i] for i in range(len(shape)-1)]
        dim = shape[-1]//2
        mean_q    = mean_log_var_q[...,:dim]
        log_var_q = mean_log_var_q[...,dim:]

        if mean_log_var_p is None:
            # assume mean=0, variance=1
            # mean_p    = 0
            # log_var_p = 0
            loss = 0.5 * (- log_var_q + K.exp(log_var_q) + K.square(mean_q) - 1)
        else:
            mean_p    = mean_log_var_p[...,:dim]
            log_var_p = mean_log_var_p[...,dim:]
            # loss = 0.5 * (log_var_p-log_var_q) + 0.5 * (K.exp(log_var_q) + K.square(mean_q-mean_p)) / K.exp(log_var_p) - 0.5
            # optimized
            loss = 0.5 * ((log_var_p-log_var_q) + (K.exp(log_var_q) + K.square(mean_q-mean_p)) / K.exp(log_var_p) - 1)

        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def call(self, mean_log_var_q):
        Gaussian.count += 1
        c = Gaussian.count-1
        layer = Lambda(self.sample,name="gaussian_{}".format(c))
        return layer(mean_log_var_q)

class Uniform(Variational):
    count = 0

    def sample(self, mean_width_q):
        sym_shape = K.shape(mean_width_q)
        shape = K.int_shape(mean_width_q)
        dims = [sym_shape[i] for i in range(len(shape)-1)]
        dim = shape[-1]//2
        mean_q = mean_width_q[...,:dim]
        width_q = mean_width_q[...,dim:]
        noise = width_q * K.random_uniform(shape=(*dims, dim),minval=-0.5, maxval=0.5)
        return K.in_train_phase(mean_q + noise, mean_q)

    def loss(self, mean_width_q):
        sym_shape = K.shape(mean_width_q)
        shape = K.int_shape(mean_width_q)
        dims = [sym_shape[i] for i in range(len(shape)-1)]
        dim = shape[-1]//2
        mean_q = mean_width_q[...,:dim]
        width_q = mean_width_q[...,dim:]

        # KL
        high_q = mean_q + width_q/2
        low_q  = mean_q - width_q/2
        high_q = K.clip(high_q, 0.0, 1.0)
        low_q  = K.clip(low_q,  0.0, 1.0)
        loss = high_q-low_q

        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def call(self, mean_width_q):
        Uniform.count += 1
        c = Uniform.count-1
        layer = Lambda(self.sample,name="uniform_{}".format(c))
        return layer(mean_width_q)

class ScheduledVariable:
    """General variable which is changed during the course of training according to some schedule"""
    def __init__(self,name="variable",):
        self.name = name
        self.variable = K.variable(self.value(0), name=name)
        
    def value(self,epoch):
        """Should return a scalar value based on the current epoch.
Each subclasses should implement a method for it."""
        pass
    
    def update(self, epoch, logs):
        K.set_value(
            self.variable,
            self.value(epoch))

class GumbelSoftmax(Variational,ScheduledVariable):
    count = 0
    
    def __init__(self,N,M,min,max,
                 annealing_start,
                 annealing_end,
                 annealer    = anneal_rate,
                 train_noise = True,
                 train_hard  = False,
                 test_noise  = False,
                 test_hard   = True, **kwargs):
        self.N           = N
        self.M           = M
        self.min         = min
        self.max         = max
        self.train_noise = train_noise
        self.train_hard  = train_hard
        self.test_noise  = test_noise
        self.test_hard   = test_hard
        self.anneal_rate = annealer(annealing_end-annealing_start,min,max)
        self.annealing_start      = annealing_start
        ScheduledVariable.__init__(self,"temperature")
        Variational.__init__(self,**kwargs)
        
    def sample(self,logit_q):
        u = K.random_uniform(K.shape(logit_q), 1e-5, 1-1e-5)
        gumbel = - K.log(-K.log(u))

        if self.train_noise:
            train_logit_q = logit_q + gumbel
        else:
            train_logit_q = logit_q
            
        if self.test_noise:
            test_logit_q = logit_q + gumbel
        else:
            test_logit_q = logit_q

        def soft_train(x):
            return K.softmax( x / self.variable )
        def hard_train(x):
            # use straight-through estimator
            argmax  = K.one_hot(K.argmax( x ), self.M)
            softmax = K.softmax( x / self.variable )
            return K.stop_gradient(argmax-softmax) + softmax
        def soft_test(x):
            return K.softmax( x / self.min )
        def hard_test(x):
            return K.one_hot(K.argmax( x ), self.M)

        if self.train_hard:
            train_activation = hard_train
        else:
            train_activation = soft_train

        if self.test_hard:
            test_activation = hard_test
        else:
            test_activation = soft_test

        return K.in_train_phase(
            train_activation( train_logit_q ),
            test_activation ( test_logit_q  ))
    
    def loss(self,logit_q,logit_p=None,p=None):
        q = K.softmax(logit_q)
        q = K.clip(q,1e-5,1-1e-5) # avoid nan in log
        q = q / K.sum(q,axis=-1,keepdims=True) # ensure sum is 1
        log_q = K.log(q)
        if (logit_p is None) and (p is None):
            # p = 1 / self.M
            # log_p = K.log(1/self.M)
            # loss = q * (log_q - log_p)
            # loss = K.sum(loss, axis=-1)
            # sum (q*logq - qlogp) = sum (q*logq) - sum (q*(-logM)) = sum qlogq + sum q logM = sum qlogq + 1*logM
            loss = K.sum(q * log_q, axis=-1) + K.log(K.cast(self.M, "float"))
        elif logit_p is not None:
            s = K.shape(logit_p)
            logit_p = wrap(logit_p, K.reshape(logit_p, (s[0], self.N, self.M)))
            p = K.softmax(logit_p)
            p = K.clip(p,1e-5,1-1e-5) # avoid nan in log
            p = p / K.sum(p,axis=-1,keepdims=True) # ensure sum is 1
            log_p = K.log(p)
            loss = q * (log_q - log_p)
            loss = K.sum(loss, axis=-1)
        elif p is not None:
            p = K.clip(p,1e-5,1-1e-5) # avoid nan in log
            # p = p / K.sum(p,axis=-1,keepdims=True) # ensure sum is 1
            log_p = K.log(p)
            loss = q * (log_q - log_p)
            loss = K.sum(loss, axis=-1)
        else:
            raise Exception("what??")

        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def call(self,logit_q):
        GumbelSoftmax.count += 1
        c = GumbelSoftmax.count-1
        s = K.shape(logit_q)
        logit_q = wrap(logit_q, K.reshape(logit_q, (s[0], self.N,self.M)))
        layer = Lambda(self.sample,name="gumbel_{}".format(c))
        return layer(logit_q)

    def value(self,epoch):
        return np.max([self.min,
                       self.max * np.exp(- self.anneal_rate * max(epoch - self.annealing_start, 0))])

class BinaryConcrete(Variational,ScheduledVariable):
    """BinaryConcrete variational layer.

Call this instance with a logit log_q (log probability),
then it adds the KL divergence loss against Bernoulli(0.5).

Optionally, you can call the instance with two logits (log_q, log_p),
where the second argument represents a target / prior distribution.
In such a case, it adds the KL divergence loss KL(Bern(q)||Bern(p)).
It is useful when you want to match two distributions.
    """
    count = 0

    def __init__(self,min,max,
                 annealing_start,
                 annealing_end,
                 annealer    = anneal_rate,
                 train_noise = True,
                 train_hard  = False,
                 test_noise  = False,
                 test_hard   = True, **kwargs):
        self.min         = min
        self.max         = max
        self.train_noise = train_noise
        self.train_hard  = train_hard
        self.test_noise  = test_noise
        self.test_hard   = test_hard
        self.anneal_rate = annealer(annealing_end-annealing_start,min,max)
        self.annealing_start      = annealing_start
        ScheduledVariable.__init__(self,"temperature")
        Variational.__init__(self,**kwargs)

    def sample(self,logit_q):
        u = K.random_uniform(K.shape(logit_q), 1e-5, 1-1e-5)
        logistic = K.log(u) - K.log(1 - u)

        if self.train_noise:
            train_logit_q = logit_q + logistic
        else:
            train_logit_q = logit_q

        if self.test_noise:
            test_logit_q = logit_q + logistic
        else:
            test_logit_q = logit_q

        def soft_train(x):
            return K.sigmoid( x / self.variable )
        def hard_train(x):
            # use straight-through estimator
            sigmoid = K.sigmoid(x / self.variable )
            step    = K.round(sigmoid)
            return K.stop_gradient(step-sigmoid) + sigmoid
        def soft_test(x):
            return K.sigmoid( x / self.min )
        def hard_test(x):
            sigmoid = K.sigmoid(x / self.min )
            return K.round(sigmoid)

        if self.train_hard:
            train_activation = hard_train
        else:
            train_activation = soft_train

        if self.test_hard:
            test_activation = hard_test
        else:
            test_activation = soft_test

        return K.in_train_phase(
            train_activation( train_logit_q ),
            test_activation ( test_logit_q  ))

    def loss(self,logit_q,logit_p=None,p=None):
        q = K.sigmoid(logit_q)
        q = K.clip(q,1e-5,1-1e-5) # avoid nan in log
        q0 = q
        q1 = 1-q
        log_q0 = K.log(q0)
        log_q1 = K.log(q1)

        if (logit_p is None) and (p is None):
            # p0 = 0.5
            # p1 = 0.5
            # log_p0 = K.log(0.5)
            # log_p1 = K.log(0.5)
            # loss = q0 * (log_q0-log_p0) + q1 * (log_q1-log_p1)
            # since -q0*log_p0 -q1*log_p1 = -q0*(log1/2) -q1*(log1/2) = -(q0+q1)*(log1/2) = 1*log2 = log2
            loss = q0 * log_q0 + q1 * log_q1 + K.log(2.0)
        elif logit_p is not None:
            p = K.sigmoid(logit_p)
            p = K.clip(p,1e-5,1-1e-5) # avoid nan in log
            p0 = p
            p1 = 1-p
            log_p0 = K.log(p0)
            log_p1 = K.log(p1)
            loss = q0 * (log_q0-log_p0) + q1 * (log_q1-log_p1)
        elif p is not None:
            p = K.clip(p,1e-5,1-1e-5) # avoid nan in log
            p0 = p
            p1 = 1-p
            log_p0 = K.log(p0)
            log_p1 = K.log(p1)
            loss = q0 * (log_q0-log_p0) + q1 * (log_q1-log_p1)
        else:
            raise Exception("what??")
        
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def call(self,logit_q):
        BinaryConcrete.count += 1
        c = BinaryConcrete.count-1
        layer = Lambda(self.sample,name="concrete_{}".format(c))
        return layer(logit_q)

    def value(self,epoch):
        return np.max([self.min,
                       self.max * np.exp(- self.anneal_rate * max(epoch - self.annealing_start, 0))])


class RandomLogistic(Initializer):
    """Initializer that generates tensors with a normal distribution.

    # Arguments
        mean: a python scalar or a scalar tensor. Mean of the random values
          to generate.
        scale: a python scalar or a scalar tensor. Scale of the
          random values to generate.
        seed: A Python integer. Used to seed the random generator.
    """

    def __init__(self, mean=0., scale=1.0, seed=None):
        self.mean = mean
        self.scale = scale
        self.seed = seed

    def __call__(self, shape, dtype=None):
        # self.mean, self.scale,
        u = K.random_uniform(shape, dtype=dtype, seed=self.seed)
        # it does log(u / 1-u)
        M, eps = tf.float32.max, tf.float32.min
        Mu = M * u
        Mu = K.clip(Mu, eps, M-eps)
        W = K.log(Mu)-K.log(M-Mu)
        return W * self.scale + self.mean

    def get_config(self):
        return {
            'mean': self.mean,
            'scale': self.scale,
            'seed': self.seed
        }

# modified from https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation
# note: this is useful only for convolutional embedding whre the same embedding
# can be reused across cells
class VQVAELayer(Layer):
    def __init__(self, embedding_dim, num_classes=2, beta=1.0,
                 initializer='uniform', epsilon=1e-10, **kwargs):
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.beta = beta
        self.initializer = keras.initializers.VarianceScaling(distribution=initializer)
        super(VQVAELayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add embedding weights.
        self.w = self.add_weight(name='embedding',
                                  shape=(self.embedding_dim, self.num_classes),
                                  initializer=self.initializer,
                                  trainable=True)
        # Finalize building.
        super(VQVAELayer, self).build(input_shape)

    def call(self, x):
        # Flatten input except for last dimension.
        flat_inputs = K.reshape(x, (-1, self.embedding_dim))

        # Calculate distances of input to embedding vectors.
        distances = (K.sum(flat_inputs**2, axis=1, keepdims=True)
                     - 2 * K.dot(flat_inputs, self.w)
                     + K.sum(self.w ** 2, axis=0, keepdims=True))

        # Retrieve encoding indices.
        encoding_indices = K.argmax(-distances, axis=1)
        encodings = K.one_hot(encoding_indices, self.num_classes)
        encoding_indices = K.reshape(encoding_indices, K.shape(x)[:-1])
        quantized = self.quantize(encoding_indices)

        e_latent_loss = K.mean((K.stop_gradient(quantized) - x) ** 2)
        q_latent_loss = K.mean((quantized - K.stop_gradient(x)) ** 2)
        self.add_loss(e_latent_loss + q_latent_loss * self.beta)

        return K.stop_gradient(quantized - x) + x

    @property
    def embeddings(self):
        return self.w

    def quantize(self, encoding_indices):
        w = K.transpose(self.embeddings.read_value())
        return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)


class BaseSchedule(ScheduledVariable):
    def __init__(self,schedule={0:0},*args,**kwargs):
        self.schedule = schedule
        super().__init__(*args,**kwargs)

class StepSchedule(BaseSchedule):
    """
       ______
       |
       |
   ____|

"""
    def __init__(self,*args,**kwargs):
        self.current_value = None
        super().__init__(*args,**kwargs)

    def value(self,epoch):
        assert epoch >= 0

        def report(value):
            if self.current_value != value:
                print(f"Epoch {epoch} StepSchedule(name={self.name}): {self.current_value} -> {value}")
                self.current_value = value
            return value

        pkey = None
        pvalue = None
        for key, value in sorted(self.schedule.items(),reverse=True):
            # from large to small
            key = int(key) # for when restoring from the json file
            if key <= epoch:
                return report(value)
            else:               # epoch < key 
                pkey, pvalue = key, value

        return report(pvalue)

class LinearSchedule(BaseSchedule):
    """
          ______
         /
        /
   ____/

"""
    def value(self,epoch):
        assert epoch >= 0
        pkey = None
        pvalue = None
        for key, value in sorted(self.schedule.items(),reverse=True):
            # from large to small
            key = int(key) # for when restoring from the json file
            if key <= epoch:
                if pkey is None:
                    return value
                else:
                    return \
                        pvalue + \
                        ( epoch - pkey ) * ( value - pvalue ) / ( key - pkey )
            else:               # epoch < key 
                pkey, pvalue = key, value

        return pvalue



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
    

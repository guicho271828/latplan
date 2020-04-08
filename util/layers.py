import keras.backend as K
from keras.layers import *
import numpy as np

debug = False
# debug = True

def Print():
    def printer(x):
        print(x)
        return x
    return Lambda(printer)

from functools import reduce
def Sequential (array):
    def apply1(arg,f):
        if debug:
            print("applying {}({})".format(f,arg))
        result = f(arg)
        if debug:
            print(K.int_shape(result), K.shape(result))
        return result
    return lambda x: reduce(apply1, array, x)

def ConditionalSequential (array, condition, **kwargs):
    def apply1(arg,f):
        if debug:
            print("applying {}({})".format(f,arg))
        concat = Concatenate(**kwargs)([flatten(condition), flatten(arg)])
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

def flatten(x):
    if K.ndim(x) >= 3:
        try:
            # it sometimes fails to infer shapes
            return Reshape((int(np.prod(K.int_shape(x)[1:])),))(x)
        except:
            return Flatten()(x)
    else:
        return x

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
class GradientEarlyStopping(Callback):
    def __init__(self, monitor='val_loss',
                 min_grad=-0.0001, epoch=1, verbose=0, smooth=3):
        super(GradientEarlyStopping, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.min_grad = min_grad
        self.history = []
        self.epoch = epoch
        self.stopped_epoch = 0
        assert epoch >= 2
        if epoch > smooth*2:
            self.smooth = smooth
        else:
            print("epoch is too small for smoothing!")
            self.smooth = epoch//2

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

    def gradient(self):
        h = np.array(self.history)
        
        # e.g. when smooth = 3, take the first/last 3 elements, average them over 3,
        # take the difference, then divide them by the epoch(== length of the history)
        return (h[-self.smooth:] - h[:self.smooth]).mean()/self.epoch
        
    def on_epoch_end(self, epoch, logs=None):
        import warnings
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        self.history.append(current) # to the last
        if len(self.history) > self.epoch:
            self.history.pop(0) # from the front
            if self.gradient() >= self.min_grad:
                self.model.stop_training = True
                self.stopped_epoch = epoch
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('\nEpoch %05d: early stopping' % (self.stopped_epoch))
            print('history:',self.history)
            print('min_grad:',self.min_grad,"gradient:",self.gradient())
    
def anneal_rate(epoch,min=0.1,max=5.0):
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

def delay(self, x, amount):
    switch = K.variable(0)
    def fn(epoch,log):
        if epoch > amount:
            K.set_value(switch, 1)
        else:
            K.set_value(switch, 0)
    self.callbacks.append(LambdaCallback(on_epoch_end=fn))
    return switch * x

def dmerge(x1, x2):
    return concatenate([wrap(x1, x1[:,None,...]),wrap(x2, x2[:,None,...])],axis=1)

def dapply(x,fn):
    x1 = wrap(x,x[:,0,...])
    x2 = wrap(x,x[:,1,...])
    y1 = fn(x1)
    y2 = fn(x2)
    y = dmerge(y1, y2)
    return y, y1, y2

class Gaussian:
    count = 0
    
    def __init__(self, beta=1.):
        self.beta = beta
        
    def call(self, mean_log_var):
        batch = K.shape(mean_log_var)[0]
        dim = K.int_shape(mean_log_var)[1]//2
        print(batch,dim)
        mean    = mean_log_var[:,:dim]
        log_var = mean_log_var[:,dim:]
        return mean + K.exp(0.5 * log_var) * K.random_normal(shape=(batch, dim))
    
    def __call__(self, mean_log_var):
        Gaussian.count += 1
        c = Gaussian.count-1

        layer = Lambda(self.call,name="gaussian_{}".format(c))

        batch = K.shape(mean_log_var)[0]
        dim = K.int_shape(mean_log_var)[1]//2
        mean    = mean_log_var[:,:dim]
        log_var = mean_log_var[:,dim:]
        loss = -0.5 * K.mean(K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis=-1)) * self.beta

        layer.add_loss(K.in_train_phase(loss, 0.0), mean_log_var)
        
        return layer(mean_log_var)

class ScheduledVariable:
    """General variable which is changed during the course of training according to some schedule"""
    def __init__(self,name="variable",):
        self.variable = K.variable(self.value(0), name=name)
        
    def value(self,epoch):
        """Should return a scalar value based on the current epoch.
Each subclasses should implement a method for it."""
        pass
    
    def update(self, epoch, logs):
        K.set_value(
            self.variable,
            self.value(epoch))

class GumbelSoftmax(ScheduledVariable):
    count = 0
    
    def __init__(self,N,M,min,max,full_epoch,annealer=anneal_rate, alpha=1., offset=0,
                 train_gumbel=True, test_gumbel=True, test_softmax=True, ):
        self.N = N
        self.M = M
        self.min = min
        self.max = max
        self.train_gumbel = train_gumbel
        self.test_gumbel = test_gumbel
        self.test_softmax = test_softmax
        self.anneal_rate = annealer(full_epoch-offset,min,max)
        self.offset = offset
        self.alpha = alpha
        super(GumbelSoftmax, self).__init__("temperature")
        
    def call(self,logits):
        u = K.random_uniform(K.shape(logits), 0, 1)
        gumbel = - K.log(-K.log(u + 1e-20) + 1e-20)

        if self.train_gumbel:
            train_logit = logits + gumbel
        else:
            train_logit = logits
            
        if self.test_gumbel:
            test_logit = logits + gumbel
        else:
            test_logit = logits

        def softmax_train(x):
            return K.softmax( x / self.variable )
        def softmax_test(x):
            return K.softmax( x / self.min )
        def argmax(x):
            return K.one_hot(K.argmax( x ), self.M)
            
        train_activation = softmax_train
        if self.test_softmax:
            test_activation = softmax_test
        else:
            test_activation = argmax
        
        return K.in_train_phase(
            train_activation( train_logit ),
            test_activation ( test_logit  ))
    
    def __call__(self,prev):
        GumbelSoftmax.count += 1
        c = GumbelSoftmax.count-1

        layer = Lambda(self.call,name="gumbel_{}".format(c))

        logits = Reshape((self.N,self.M))(prev)
        q = K.softmax(logits)
        log_q = K.log(q + 1e-20)
        loss = K.mean(q * log_q) * self.alpha

        layer.add_loss(K.in_train_phase(loss, 0.0), logits)

        return layer(logits)

    def value(self,epoch):
        return np.max([self.min,
                       self.max * np.exp(- self.anneal_rate * max(epoch - self.offset, 0))])

class BaseSchedule(ScheduledVariable):
    def __init__(self,schedule={0:0}):
        self.schedule = schedule
        super(BaseSchedule, self).__init__()

class StepSchedule(BaseSchedule):
    """
       ______
       |
       |
   ____|

"""
    def value(self,epoch):
        assert epoch >= 0
        pkey = None
        pvalue = None
        for key, value in sorted(self.schedule.items(),reverse=True):
            # from large to small
            key = int(key) # for when restoring from the json file
            if key <= epoch:
                return value
            else:               # epoch < key 
                pkey, pvalue = key, value

        return pvalue

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

# modified version
import progressbar
class DynamicMessage(progressbar.DynamicMessage):
    def __call__(self, progress, data):
        val = data['dynamic_messages'][self.name]
        if val:
            return self.name + ': ' + '{}'.format(val)
        else:
            return self.name + ': ' + 6 * '-'



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
    

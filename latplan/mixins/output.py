import keras.backend.tensorflow_backend as K
from keras.layers import Activation, Lambda
from latplan.util.distances import *
from latplan.util.tuning    import InvalidHyperparameterError
from latplan.util.layers    import StepSchedule, LinearSchedule
from latplan.puzzles.objutil import *
import math
import numpy as np
from skimage.transform import resize
from skimage.util      import img_as_ubyte

# output activations and loss function #########################################


# ###### HACK! HACK! HACK! ######
# 
# self.parameters is *** set *** by EncoderDecoderMixin in mixin/encoder_decoder.py .
# this is an ugly solution to LocationOutput which requres the size of image dimensions.
# Later, this is also used for normalize/unnormalize, which used to take external parameters.

class VanillaRenderer:
    def normalize(self,x):
        # Note: this method is later used by the planner. It is not used during the training/plotting.
        mean = np.array(self.parameters["mean"])
        std  = np.array(self.parameters["std"])
        return (x-mean)/(std+1e-20)

    def unnormalize(self,x):
        mean = np.array(self.parameters["mean"])
        std  = np.array(self.parameters["std"])
        return (x*std)+mean

    def render(self,x,*args,**kwargs):
        return self.unnormalize(x)


class GaussianOutput(VanillaRenderer):
    def __init__(self,sigma=0.1,eval_sigma=None):
        self._sigma = sigma
        if eval_sigma is None:
            self._eval_sigma = sigma
        else:
            self._eval_sigma = eval_sigma
    def loss(self, true, pred):
        # Gaussian distribution N(x|y, sigma) is exp [- (x-y)^2 / (2*sigma^2)] / sqrt(2*pi*sigma^2)
        # negative log probability that the prediction y follows N(x, sigma) is
        # (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2
        sigma = K.in_train_phase(self._sigma, self._eval_sigma)
        v1 = 2*sigma*sigma
        v2 = 2*math.pi*sigma*sigma

        loss = K.square(true - pred) / v1 # + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")


class StepScheduleGaussianOutput(StepSchedule,VanillaRenderer):
    def loss(self, true, pred):
        # negative log probability that the prediction y follows N(x, sigma) is
        # (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2

        # use the sharpest sigma in the schedule
        sigma = K.in_train_phase(self.variable, min(self.schedule.values()))
        v1 = 2*sigma*sigma
        v2 = 2*math.pi*sigma*sigma

        loss = K.square(true - pred) / v1 # + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")


class LinearScheduleGaussianOutput(LinearSchedule,VanillaRenderer):
    def loss(self, true, pred):
        # negative log probability that the prediction y follows N(x, sigma) is
        # (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2

        # use the sharpest sigma in the schedule
        sigma = K.in_train_phase(self.variable, min(self.schedule.values()))
        v1 = 2*sigma*sigma
        v2 = 2*math.pi*sigma*sigma

        loss = K.square(true - pred) / v1 # + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")


class BayesGaussianOutput(VanillaRenderer):
    "The prediction has twice the channel size of the true data."
    def __init__(self):
        pass
    def loss(self, true, pred):
        # negative log probability that the prediction y follows N(x, sigma) is
        # (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2
        # (x-y)^2 / 2var  + log (2*pi*var) / 2
        # (x-y)^2 / 2 exp(log_var)  + (log (2*pi)+ log_var) / 2
        shape = K.int_shape(pred)
        dim = shape[-1]//2
        mean    = pred[...,:dim]
        log_var = pred[...,dim:]
        loss = K.square(true - pred) / (2*K.exp(log_var)) + (K.log(2*pi)+log_var)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")


class PNormOutput(VanillaRenderer):
    """I do not know what this loss function correspond to."""
    def __init__(self,sigma=0.1,p=4):
        self._sigma = sigma
        self.p = p
    def loss(self, true, pred):
        loss = ((true - pred) / self._sigma) ** self.p
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")


class SharedSigmaGaussianOutput(VanillaRenderer):
    """Learns a single sigma shared across the data points and output dimensions."""
    def __init__(self):
        self.logsigma = K.softplus(K.variable(0.0)+6)-6 # sigma=1.0, with soft clipping at -6

    def loss(self, true, pred):
        # negative log probability that the prediction y follows N(x, sigma) is
        # (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2
        sigma_square = K.exp(2*self.logsigma)
        v1 = 2*sigma_square
        v2 = 2*math.pi*sigma_square

        loss = K.square(true - pred) / v1 + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return K.exp(self.logsigma)

    def activation(self):
        return Activation("linear")


class OptimalSigmaGaussianOutput(VanillaRenderer):
    """Uses an analytically obtained sigma that maximizes the negative log likelihood of Gaussian.
Axes specify which dimension to share the sigma across.
For example:
 axis=0 means it uses a different sigma for a different pixel, but the these sigmas are shared across the dataset.
 axis=[1,2,3] for images means it uses a different sigma for each image, and the sigma is shared across pixels.
              This corresponds to the per-image optimal sigma in the paper (Rybkin et al, 2020)
 axis=None means it uses a shared sigma across the dataset and the pixels.
"""
    def __init__(self,axis=None):
        self.axis = axis

    def loss(self, true, pred):
        # L = (x-y)^2 / 2sigma^2  + log (2*pi*sigma^2) / 2
        #   = (x-y)^2 / 2sigma^2  + log sigma + log (2*pi) / 2
        # sigma* = argmin_sigma L
        # Differentiated by sigma,
        # dL/dsigma = -2 (x-y)^2 / 2sigma^3  + 1 / sigma = 0
        # 1/sigma = (x-y)^2 / sigma^3
        # sigma^2 = (x-y)^2
        square = K.square(true - pred)
        sigma_square = K.maximum(K.mean(square, axis=self.axis, keepdims=True), K.epsilon())
        v1 = 2*sigma_square
        v2 = 2*math.pi*sigma_square

        loss = square / v1 + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        square = K.square(true - pred)
        sigma_square = K.mean(square, axis=self.axis, keepdims=True)
        return K.sqrt(sigma_square)

    def activation(self):
        return Activation("linear")


class ProbabilityOutput(VanillaRenderer):

    def loss(self, true, pred):
        # true dataset is normalized as if it is a gaussian dataset. unnormalize into 0-1 range
        if not hasattr(self, "__mean"):
            self.__mean = K.constant(self.parameters["mean"])
            self.__std  = K.constant(self.parameters["std"])
        true_binary = (true*self.__std)+self.__mean
        loss = K.binary_crossentropy(true_binary, pred)
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def activation(self):
        return Activation("sigmoid")


class SinusoidalOutput(VanillaRenderer):
    def loss(self, true, pred):
        # -1 < x < 1, and x = sin(theta).
        # cosine distance / dot product can be thought of as
        # negative log likelihood of a spherical distribution.
        # For x = cos(theta),
        # dot product xy = cos(t1)cos(t2) = log p
        # p = exp(-cos(t1)cos(t2)) / C
        # where C is a normalizing constant (because the integral must be 1 to be a probability distribution).
        # 
        # \int_0^{2pi} exp(cos(t)) dt = 2pi I_0 (1) ~ 7.95 .
        # where I_0 is a modified Bessel function of the first kind.
        #
        # cf. Hyperspherical Variational Auto-Encoders Tim R. Davidson Luca Falorsi Nicola De Cao Thomas Kipf Jakub M. Tomczak
        # 
        # Unfortunately, because I dont have time and suck at math I omit the normalizing constant.
        loss = K.sum(- true * pred, axis=-1)
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def activation(self):
        # the input is an angle.
        # the even axes are cos(x) in the dataset generation,
        # but it does not matter because cos(x) = sin(x+pi/2).
        return Lambda(lambda x: K.sin(x))


class CategoricalOutput(VanillaRenderer):
    def loss(self, true, pred):
        loss = K.categorical_crossentropy(true, pred)
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def activation(self):
        return Activation("softmax")


class LaplacianOutput(VanillaRenderer):
    def __init__(self,sigma=0.1,eval_sigma=None):
        self._sigma = sigma
        if eval_sigma is None:
            self._eval_sigma = sigma
        else:
            self._eval_sigma = eval_sigma
    def loss(self, true, pred):
        # Laplace distribution L(x|y, sigma) is exp [- |x-y|/sigma] / 2sigma
        # its negative log probability i
        # |x-y| / 2sigma + log 2sigma
        sigma = K.in_train_phase(self._sigma, self._eval_sigma)
        v1 = 2*sigma
        v2 = 2*sigma

        loss = K.abs(true - pred) / v1 # + K.log(v2)/2
        # sum across dimensions
        loss = K.batch_flatten(loss)
        loss = K.sum(loss, axis=-1)
        return loss

    def sigma(self, true, pred):
        return self.sigma

    def activation(self):
        return Activation("linear")




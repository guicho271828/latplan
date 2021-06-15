import keras
from keras.layers import *
from keras.layers.normalization import BatchNormalization as BN
from latplan.util.distances import *
from latplan.util.layers    import *
from latplan.util.perminv   import *
from latplan.util.tuning    import InvalidHyperparameterError
from latplan.mixins.output import *

# Encoder / Decoder ################################################################

# Implementation notes:
#
# All encoder starts with a noise layer and a BN layer.
# In the intermediate layers,
# BN comes before Dropout; there is no particular reason, but I use this.
#
# All decoder starts with an optional dropout (disabled by default) and a BN layer.

def output_shape(layers,input_shape):
    """Utility for computing the output shape of a list of layers."""
    from functools import reduce
    def c(input_shape,layer):
        print(layer)
        shape = layer.compute_output_shape(input_shape)
        print(input_shape, "->", shape, ":", layer)
        return shape
    return reduce(c,layers,input_shape)


class EncoderDecoderMixin:
    def _build_around(self,state_input_shape):
        self.output = eval(self.parameters["output"])
        self.output.parameters = self.parameters
        if hasattr(self.output, "update"):
            from keras.callbacks import LambdaCallback
            self.callbacks.append(LambdaCallback(on_epoch_end=self.output.update))
        self.encoder_net = self.build_encoder(state_input_shape)
        self.decoder_net = self.build_decoder(state_input_shape)
        super()._build_around(state_input_shape)
        return
    def _encode(self,x):
        return Sequential(self.encoder_net)(x)
    def _decode(self,x):
        return Sequential(self.decoder_net)(x)
    def build_encoder(self,input_shape):
        return [
            GaussianNoise(self.parameters["noise"]),
            BN(),
            *self._build_encoder(input_shape),
            self.activation(),
        ]
    def build_decoder(self,input_shape):
        return [
            BN(),
            *([Dropout(self.parameters["dropout"])] if self.parameters["dropout_z"] else []),
            *self._build_decoder(input_shape),
            Reshape(input_shape),
            self.output.activation()
        ]
    pass


class FullConnectedMixin(EncoderDecoderMixin):
    def _build_encoder(self,input_shape):
        return [MyFlatten(),
                *[keras.Sequential([
                    Dense(self.parameters["fc_width"], activation="relu", use_bias=False, kernel_initializer="he_uniform"),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                  for _ in range(self.parameters["fc_depth"]-1)],
                Dense(np.prod(self.zindim())),
        ]

    def _build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [MyFlatten(),
                *[keras.Sequential([
                    Dense(self.parameters["fc_width"], activation="relu", use_bias=False, kernel_initializer="he_uniform"),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                  for _ in range(self.parameters["fc_depth"]-1)],
                Dense(data_dim),
        ]

    pass


class ConvolutionalMixin(EncoderDecoderMixin):
    """A mixin that uses only convolutional layers in the encoder/decoder.
When self.parameters["N"] is specified, it wraps the latent layer with a Dense layer."""
    def _build_around(self,input_shape):
        self.has_dense_layer = ("N" in self.parameters)
        super()._build_around(input_shape)

    def autocrop_dimensions(self,mod_input_shape,d):
        p  = self.parameters["conv_pooling"]
        total_pool = p ** d
        H, W, C = mod_input_shape
        import math
        dH = math.ceil(H/total_pool)*total_pool - H
        dW = math.ceil(W/total_pool)*total_pool - W
        print("pool per layer:",p,"depth:",d,"total pool:",total_pool,"H:",H,"W:",W,"dH:",dH,"dW:",dW)
        return dH, dW

    def _build_encoder(self,input_shape):
        print("building a convolutional encoder")
        layers = []

        if self.has_dense_layer:
            d = self.parameters["conv_depth"]-1
        else:
            d = self.parameters["conv_depth"]

        dH, dW = self.autocrop_dimensions(input_shape, d)
        if dH !=0 or dW!=0:
            layers.append(ZeroPadding2D(padding=((0,dH),(0,dW),)))

        for i in range(d):
            layers.extend(self.encoder_block(i))

        self.conv_latent_space = output_shape(layers,[0,*input_shape])[1:] # H,W,C
        flat_size = np.prod(self.conv_latent_space)
        layers.append(
            # MyFlatten(),
            Reshape((flat_size,)))
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        # Conv2D does not set the tensor shape properly, and MyFlatten() fails to work
        # during the build_aux phase.

        if self.has_dense_layer:
            layers.append(Dense(self.parameters["N"]))
        else:
            self.parameters["N"] = flat_size

        print(f"latent space shape is {self.conv_latent_space} : {self.parameters['N']} propositions in total")

        return layers

    def _build_decoder(self,input_shape):
        print("building a convolutional decoder")
        layers = []
        if self.has_dense_layer:
            flat_size = np.prod(self.conv_latent_space)
            layers.append(Dense(flat_size))
            layers.append(BN())
            layers.append(Dropout(self.parameters["dropout"]))

        layers.append(Reshape(self.conv_latent_space))

        if self.has_dense_layer:
            d = self.parameters["conv_depth"]-1
        else:
            d = self.parameters["conv_depth"]
        for i in range(d-1, 0, -1):
            layers.extend(self.decoder_block(i, input_shape))

        layers.extend(self.decoder_block(0, input_shape))

        dH, dW = self.autocrop_dimensions(input_shape, d)
        if dH !=0 or dW!=0:
            layers.append(Cropping2D(cropping=((0,dH),(0,dW),)))

        if self.has_dense_layer:
            computed_input_shape = output_shape(layers[3:],[0,*self.conv_latent_space])[1:]
        else:
            computed_input_shape = output_shape(layers,[0,*self.conv_latent_space])[1:]
        assert input_shape == computed_input_shape

        return layers

    pass


class MaxPoolingConvolutionalMixin(ConvolutionalMixin):
    def encoder_block(self,i):
        """Extend this method for Residual Nets"""
        k = self.parameters["conv_kernel"]
        p = self.parameters["conv_pooling"]
        w  = self.parameters["conv_channel"]
        dw = self.parameters["conv_channel_increment"]
        cpp = self.parameters["conv_per_pooling"]
        layers = []
        for j in range(cpp):
            layers.extend([
                Convolution2D(round(w * (dw ** i)), k, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform"),
                BN(),
                Dropout(self.parameters["dropout"]),
            ])
        if p > 1:
            layers.append(MaxPooling2D((p,p)))
        return layers

    def decoder_block(self,i,input_shape):
        """Extend this method for Residual Nets"""
        k = self.parameters["conv_kernel"]
        p = self.parameters["conv_pooling"]
        w  = self.parameters["conv_channel"]
        dw = self.parameters["conv_channel_increment"]
        cpp = self.parameters["conv_per_pooling"]
        layers = []
        if p > 1:
            layers.append(UpSampling2D((p,p)))
        for j in range(cpp):
            if i == 0 and j == cpp-1: # last layer
                # no activation, batchnorm, dropout. use bias
                layers.append(
                    Deconvolution2D(input_shape[-1],k, padding="same"))
            else:
                layers.extend([
                    Deconvolution2D(round(w * (dw ** i)),k, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform"),
                    BN(),
                    Dropout(self.parameters["dropout"]),
                ])
        return layers


class StridedConvolutionalMixin(ConvolutionalMixin):
    def encoder_block(self,i):
        """Extend this method for Residual Nets"""
        k = self.parameters["conv_kernel"]
        p = self.parameters["conv_pooling"]
        w  = self.parameters["conv_channel"]
        dw = self.parameters["conv_channel_increment"]
        cpp = self.parameters["conv_per_pooling"]
        layers = []
        for j in range(cpp):
            if j == 0:
                conv = Convolution2D(round(w * (dw ** i)), k, strides=p, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform")
            else:
                conv = Convolution2D(round(w * (dw ** i)), k, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform")
            layers.extend([
                conv,
                BN(),
                Dropout(self.parameters["dropout"]),
            ])
        return layers

    def decoder_block(self,i,input_shape):
        """Extend this method for Residual Nets"""
        k = self.parameters["conv_kernel"]
        p = self.parameters["conv_pooling"]
        w  = self.parameters["conv_channel"]
        dw = self.parameters["conv_channel_increment"]
        cpp = self.parameters["conv_per_pooling"]
        layers = []
        for j in range(cpp):
            if j == cpp-1:
                if i == 0:      # outermost layer
                    # no activation, batchnorm, dropout. use bias
                    layers.append(
                        Deconvolution2D(input_shape[-1],k, strides=p, padding="same"))
                else:
                    layers.extend([
                        Deconvolution2D(round(w * (dw ** i)), k, strides=p, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform"),
                        BN(),
                        Dropout(self.parameters["dropout"]),
                    ])
            else:
                layers.extend([
                    Deconvolution2D(round(w * (dw ** i)), k, activation="relu", padding="same", use_bias=False, kernel_initializer="he_uniform"),
                    BN(),
                    Dropout(self.parameters["dropout"]),
                ])
        return layers



class GaussianDecoderMixin(EncoderDecoderMixin):
    "Provides a decoder whose output channel size is twice the size of the input. Compatible with mixins.output.BayesGaussianOutput"
    def build_decoder(self,input_shape):
        channel_dims = input_shape[-1]
        return [
            BN(),
            *([Dropout(self.parameters["dropout"])] if self.parameters["dropout_z"] else []),
            *self._build_decoder(input_shape),
            Reshape(input_shape[:-1]+[channel_dims*2]),
            self.output.activation()
        ]


class FullConnectedGaussianMixin(GaussianDecoderMixin,FullConnectedMixin):
    def _build_decoder(self,input_shape):
        data_dim = np.prod(input_shape)
        return [MyFlatten(),
                *[keras.Sequential([
                    Dense(self.parameters["fc_width"], activation="relu", use_bias=False, kernel_initializer="he_uniform"),
                    BN(),
                    Dropout(self.parameters["dropout"]),])
                  for _ in range(self.parameters["fc_depth"]-1)],
                Dense(data_dim*2),
        ]

    pass


class GaussianConvolutionalMixin(ConvolutionalMixin):
    def _build_decoder(self,input_shape):
        print("building a convolutional decoder")
        layers = []
        if self.has_dense_layer:
            flat_size = np.prod(self.conv_latent_space)
            layers.append(Dense(flat_size))
            layers.append(BN())
            layers.append(Dropout(self.parameters["dropout"]))

        layers.append(Reshape(self.conv_latent_space))

        if self.has_dense_layer:
            d = self.parameters["conv_depth"]-1
        else:
            d = self.parameters["conv_depth"]
        for i in range(d-1, 0, -1):
            layers.extend(self.decoder_block(i, input_shape))

        channel_dims = input_shape[-1]
        layers.extend(self.decoder_block(0, (input_shape[:-1]+[channel_dims*2])))

        dH, dW = self.autocrop_dimensions(input_shape, d)
        if dH !=0 or dW!=0:
            layers.append(Cropping2D(cropping=((0,dH),(0,dW),)))

        if self.has_dense_layer:
            computed_input_shape = output_shape(layers[3:],[0,*self.conv_latent_space])[1:]
        else:
            computed_input_shape = output_shape(layers,[0,*self.conv_latent_space])[1:]
        assert input_shape == computed_input_shape

        return layers

    pass

class GaussianMaxPoolingConvolutionalMixin(GaussianConvolutionalMixin,MaxPoolingConvolutionalMixin):
    pass

class GaussianStridedConvolutionalMixin(GaussianConvolutionalMixin,StridedConvolutionalMixin):
    pass



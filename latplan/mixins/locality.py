from keras.layers import *
from keras.layers.normalization import BatchNormalization as BN
from latplan.util.distances import *
from latplan.util.layers    import *
from latplan.util.perminv   import *
from latplan.util.tuning    import InvalidHyperparameterError

# Locality regularization ################################################################
#
# Apply a regularization for TransitionAE that minimize the difference between two latent states,
# assuming that the difference is small.

# Note 2020/12/22:
# Currently abandoned, as it was not successful in the early research (2019 Summer).
# It is, however, before the discovery of cube-space prior, thus this is worth revisiting in the future.

class LocalityMixin:
    def _build(self,input_shape):
        super()._build(input_shape)
        self.locality_alpha = StepSchedule(schedule={
            0:0,
            (self.parameters["epoch"]*self.parameters["locality_delay"]):self.parameters["locality"],
        }, name="locality")
        self.callbacks.append(LambdaCallback(on_epoch_end=self.locality_alpha.update))

        def locality(x, y):
            return self.locality_alpha.variable

        self.metrics.append(locality)
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


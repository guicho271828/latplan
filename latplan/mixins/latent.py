from latplan.util.distances import *
from latplan.util.layers    import *
from latplan.util.perminv   import *
from latplan.util.tuning    import InvalidHyperparameterError

# Latent Activations ################################################################

class ConcreteLatentMixin:
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
            MyFlatten(),
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
            MyFlatten(),
        ])


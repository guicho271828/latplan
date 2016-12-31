import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse

class GumbelAE:
    # common options
    min_temperature = 0.1
    max_temperature = 5.0
    anneal_rate = 0.0003
    
    def __init__(self,data_dim,M=2,N=16):
        def to2(tensor):
            return K.reshape(tensor, (-1, N, M))
        def to1(tensor):
            return K.reshape(tensor, (-1, N*M))
        
        tau = K.variable(self.max_temperature, name="temperature")
        def sampling(logits):
            U = K.random_uniform(K.shape(logits), 0, 1)
            z = logits - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            return to1(softmax(to2( z / tau )))
        
        x = Input(shape=(data_dim,))
        _encoder = Sequential([
            Dense(512, activation='relu', input_shape=(data_dim,)),
            Dense(256, activation='relu'),
            Dense(M*N),])
        logits = _encoder(x)
        z = Lambda(sampling)(logits)
        _decoder = Sequential([
            Dense(256, activation='relu', input_shape=(N*M,)),
            Dense(512, activation='relu'),
            Dense(data_dim, activation='sigmoid')])
        y = _decoder(z)

        def gumbel_loss(x, y):
            q = softmax(to2(logits))
            log_q = K.log(q + 1e-20)
            kl_tmp = q * (log_q - K.log(1.0/M))
            KL = K.sum(kl_tmp, axis=(1, 2))
            elbo = data_dim * bce(x, y) - KL
            return elbo
        
        self.M, self.N = M, N
        self.__tau = tau
        self.__loss = gumbel_loss
        self.encoder     = Model(x, z)
        self.decoder     = _decoder
        self.autoencoder = Model(x, y)

    import os.path as p
    def save(self,path):
        import subprocess
        subprocess.call(["mkdir",path])
        self.encoder.save_weights(p.join(path,"encoder.h5"))
        self.decoder.save_weights(p.join(path,"decoder.h5"))
    def load(self,path):
        self.encoder.load_weights(p.join(path,"encoder.h5"))
        self.decoder.load_weights(p.join(path,"decoder.h5"))
    def train(self,data,epoch=200,batch_size=1000,optimizer='adam'):
        try:
            self.autoencoder.compile(optimizer=optimizer, loss=self.__loss)
            for e in range(epoch):
                self.autoencoder.fit(data,data, shuffle=True, nb_epoch=1, batch_size=batch_size)
                new_tau = np.max([K.get_value(self.__tau) * np.exp(- self.anneal_rate * e),
                                  self.min_temperature])
                print "Tau = {} epoch = {}".format(new_tau,e)
                K.set_value(self.__tau, new_tau)
        except KeyboardInterrupt:
            print ("learning stopped")
        self.autoencoder.compile(optimizer=optimizer, loss=mse)
        print "MSE reconstruction error: {}".format(self.autoencoder.evaluate(data,data,verbose=0))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        print "BCE reconstruction error: {}".format(self.autoencoder.evaluate(data,data,verbose=0))
    def encode(self,data):
        return self.encoder.predict(data)
    def decode(self,data):
        return self.decoder.predict(data)
    def autoencode(self,data):
        return self.autoencoder.predict(data)
    def encode_binary(self,data):
        assert M == 2, "M={}, not 2".format(M)
        return self.encode(data).reshape(-1, N, M)[:,:,0].reshape(-1, N)
    def summary(self):
        self.autoencoder.summary()

if __name__ == '__main__':
    import subprocess
    import os.path as p
    if p.exists("test/"):
        subprocess.call("rm -rf test")
    ae = GumbelAE(784)
    ae.save("test/")
    del ae
    ae = GumbelAE(784)
    ae.load("test/")
    

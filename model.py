import numpy as np

from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model, Sequential
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.activations import softmax
from keras.objectives import binary_crossentropy as bce
from keras.objectives import mse
from keras.callbacks import LambdaCallback

class GumbelAE:
    # common options
    min_temperature = 0.1
    max_temperature = 5.0
    anneal_rate = 0.0003
    
    def __init__(self,path,M=2,N=16):
        import subprocess
        subprocess.call(["mkdir",path])
        self.path = path
        self.M, self.N = M, N
        self.built = False
        self.loaded = False
        
    def build(self,input_shape):
        if self.built:
            print "Avoided building {} twice.".format(self)
            return
        data_dim = np.prod(input_shape)
        print "input_shape:{}, flattened into {}".format(input_shape,data_dim)
        M, N = self.M, self.N
        tau = K.variable(self.max_temperature, name="temperature")
        def sampling(logits):
            U = K.random_uniform(K.shape(logits), 0, 1)
            z = logits - K.log(-K.log(U + 1e-20) + 1e-20) # logits + gumbel noise
            return softmax( z / tau )
        
        x = Input(shape=input_shape)
        _encoder = Sequential([
            Reshape((data_dim,),input_shape=input_shape),
            Dense(512, activation='relu'),
            Dense(256, activation='relu'),
            Dense(M*N),
            Reshape((N,M))])
        logits = _encoder(x)
        z = Lambda(sampling)(logits)
        _decoder = Sequential([
            Reshape((N*M,),input_shape=(N,M,)),
            Dense(256, activation='relu'),
            Dense(512, activation='relu'),
            Dense(data_dim, activation='sigmoid'),
            Reshape(input_shape),])
        y = _decoder(z)

        def gumbel_loss(x, y):
            q = softmax(logits)
            log_q = K.log(q + 1e-20)
            kl_tmp = q * (log_q - K.log(1.0/M))
            KL = K.sum(kl_tmp, axis=(1, 2))
            elbo = data_dim * bce(x, y) - KL
            return elbo
        
        self.__tau = tau
        self.__loss = gumbel_loss
        self.encoder     = Model(x, z)
        self.decoder     = _decoder
        self.autoencoder = Model(x, y)
        self.built = True
    def local(self,path):
        import os.path as p
        return p.join(self.path,path)
    def save(self):
        self.encoder.save_weights(self.local("encoder.h5"))
        self.decoder.save_weights(self.local("decoder.h5"))
    def do_load(self):
        self.encoder.load_weights(self.local("encoder.h5"))
        self.decoder.load_weights(self.local("decoder.h5"))
        self.loaded = True
    def load(self):
        if not self.loaded:
            self.do_load()
        
    def cool(self, epoch, logs):
        new_tau = np.max([K.get_value(self.__tau) * np.exp(- self.anneal_rate * epoch),
                          self.min_temperature])
        print "Tau = {}".format(new_tau)
        K.set_value(self.__tau, new_tau)
        
    def train(self,train_data,epoch=200,batch_size=1000,optimizer='adam',test_data=None,save=True):
        self.build(train_data.shape[1:])
        self.summary()
        if test_data is not None:
            validation = (test_data,test_data)
        else:
            validation = None
        try:
            self.autoencoder.compile(optimizer=optimizer, loss=self.__loss)
            self.autoencoder.fit(
                train_data, train_data,
                nb_epoch=epoch, batch_size=batch_size,
                shuffle=True, validation_data=validation,
                callbacks=[LambdaCallback(on_epoch_begin=self.cool)])
        except KeyboardInterrupt:
            print ("learning stopped")
        self.autoencoder.compile(optimizer=optimizer, loss=mse)
        print "Reconstruction MSE: {}".format(self.autoencoder.evaluate(train_data,train_data,verbose=0))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        print "Reconstruction BCE: {}".format(self.autoencoder.evaluate(train_data,train_data,verbose=0))
        self.loaded = True
        if save:
            self.save()
    def encode(self,data):
        self.build(data.shape[1:])
        self.load()
        return self.encoder.predict(data)
    def decode(self,data):
        return self.decoder.predict(data)
    def autoencode(self,data):
        self.build(data.shape[1:])
        self.load()
        return self.autoencoder.predict(data)
    def encode_binary(self,data):
        assert self.M == 2, "M={}, not 2".format(self.M)
        return self.encode(data)[:,:,0].reshape(-1, self.N)
    def decode_binary(self,data):
        assert self.M == 2, "M={}, not 2".format(self.M)
        return self.decode(np.stack((data,1-data),axis=-1))
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import shlex, subprocess
    from mnist import mnist
    x_train, _, x_test, _ = mnist()
    ae = GumbelAE("samples/mnist_model/")
    ae.train(x_train,test_data=x_test)
    ae.summary()
    del ae
    ae = GumbelAE("samples/mnist_model/")
    howmany=10
    y_test = ae.autoencode(x_test[:howmany])
    z_test = ae.encode_binary(x_test[:howmany])

    plt.figure(figsize=(30, 5))
    for i in range(howmany):
        plt.subplot(3,howmany,i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        plt.subplot(3,howmany,i+1+1*howmany)
        plt.imshow(z_test[i].reshape(4,4), cmap='gray',
                   interpolation='nearest')
        plt.axis('off')
        plt.subplot(3,howmany,i+1+2*howmany)
        plt.imshow(y_test[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.savefig(ae.local('viz.png'))



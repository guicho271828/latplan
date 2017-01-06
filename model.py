import numpy as np

from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, Activation, Cropping2D, SpatialDropout2D, Lambda, GaussianNoise
from keras.layers.normalization import BatchNormalization as BN
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
    
    def __init__(self,path,M=2,N=16):
        import subprocess
        subprocess.call(["mkdir",path])
        self.path = path
        self.M, self.N = M, N
        self.built = False
        self.loaded = False
        self.min_temperature = 0.1
        self.max_temperature = 5.0
        self.anneal_rate = 0.0003
        self.verbose = True
        
    def build(self,input_shape):
        if self.built:
            if self.verbose:
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
        def apply1(arg,f):
            return f(arg)
        x = Input(shape=input_shape)
        _encoder = [Reshape((data_dim,)),
                    GaussianNoise(0.1),
                    Dense(1000, activation='relu'),
                    Dropout(0.4),
                    BN(),
                    Dense(1000, activation='relu'),
                    Dropout(0.4),
                    BN(),
                    Dense(M*N),
                    Reshape((N,M))]
        logits = reduce(apply1, _encoder, x)
        z = Lambda(sampling)(logits)
        _decoder = [Reshape((N*M,)),
                    Dense(256, activation='relu'),
                    Dense(512, activation='relu'),
                    Dense(data_dim, activation='sigmoid'),
                    Reshape(input_shape),]
        y = reduce(apply1, _decoder, z)
        z2 = Input(shape=(N,M))
        y2 = reduce(apply1, _decoder, z2)
        z3 = Lambda(lambda z:K.round(z))(z)
        y3 = reduce(apply1, _decoder, z3)

        def gumbel_loss(x, y):
            q = softmax(logits)
            log_q = K.log(q + 1e-20)
            kl_loss = K.sum(q * (log_q - K.log(1.0/M)), axis=(1, 2))
            reconstruction_loss = data_dim * bce(K.reshape(x,(K.shape(x)[0],data_dim,)),
                                                 K.reshape(y,(K.shape(x)[0],data_dim,)))
            # l2_loss = K.sum(K.square(z[:,0]))
            # l1_loss = K.sum(z[:,0])
            # l2_loss = data_dim * K.mean(K.square(z[:,0]))
            # l1_loss = data_dim * K.mean(z[:,0])
            return reconstruction_loss - kl_loss
        
        self.__tau = tau
        self.__loss = gumbel_loss
        self.encoder     = Model(x, z)
        self.decoder     = Model(z2, y2)
        self.autoencoder = Model(x, y)
        self.autoencoder_binary = Model(x, y3)
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
        
    def train(self,train_data,
              epoch=200,batch_size=1000,optimizer='adam',test_data=None,save=True,**kwargs):
        for k,v in kwargs.iteritems():
            setattr(self, k, v)
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
        self.loaded = True
        v = self.verbose
        self.verbose = False
        def test_both(msg, fn):
            print msg.format(fn(train_data))
            if test_data is not None:
                print (msg+" (validation)").format(fn(test_data))
        self.autoencoder.compile(optimizer=optimizer, loss=mse)
        test_both("Reconstruction MSE: {}",
                  lambda (data): self.autoencoder.evaluate(data,data,verbose=0))
        self.autoencoder_binary.compile(optimizer=optimizer, loss=mse)
        test_both("Binary Reconstruction MSE: {}",
                  lambda (data): self.autoencoder_binary.evaluate(data,data,verbose=0))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        test_both("Reconstruction BCE: {}",
                  lambda (data): self.autoencoder.evaluate(data,data,verbose=0))
        self.autoencoder_binary.compile(optimizer=optimizer, loss=bce)
        test_both("Binary Reconstruction BCE: {}",
                  lambda (data): self.autoencoder_binary.evaluate(data,data,verbose=0))
        test_both("Latent activation: {}",
                  lambda (data): self.encode_binary(train_data).mean())
        self.verbose = v
        if save:
            self.save()
    def encode(self,data,**kwargs):
        self.build(data.shape[1:])
        self.load()
        return self.encoder.predict(data,**kwargs)
    def decode(self,data,**kwargs):
        return self.decoder.predict(data,**kwargs)
    def autoencode(self,data):
        self.build(data.shape[1:])
        self.load()
        return self.autoencoder.predict(data)
    def encode_binary(self,data,**kwargs):
        assert self.M == 2, "M={}, not 2".format(self.M)
        return self.encode(data,**kwargs)[:,:,0].reshape(-1, self.N)
    def decode_binary(self,data,**kwargs):
        assert self.M == 2, "M={}, not 2".format(self.M)
        return self.decode(np.stack((data,1-data),axis=-1),**kwargs)
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



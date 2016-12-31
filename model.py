import numpy as np

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
    def train(self,train_data,epoch=200,batch_size=1000,optimizer='adam',test_data=None):
        if test_data is not None:
            validation = (test_data,test_data)
        else:
            validation = None
        try:
            self.autoencoder.compile(optimizer=optimizer, loss=self.__loss)
            for e in range(epoch):
                self.autoencoder.fit(train_data, train_data,
                                     shuffle=True, nb_epoch=1, batch_size=batch_size,
                                     validation_data=validation)
                new_tau = np.max([K.get_value(self.__tau) * np.exp(- self.anneal_rate * e),
                                  self.min_temperature])
                print "Tau = {} epoch = {}".format(new_tau,e)
                K.set_value(self.__tau, new_tau)
        except KeyboardInterrupt:
            print ("learning stopped")
        self.autoencoder.compile(optimizer=optimizer, loss=mse)
        print "MSE reconstruction error: {}".format(self.autoencoder.evaluate(train_data,train_data,verbose=0))
        self.autoencoder.compile(optimizer=optimizer, loss=bce)
        print "BCE reconstruction error: {}".format(self.autoencoder.evaluate(train_data,train_data,verbose=0))
    def encode(self,data):
        return self.encoder.predict(data)
    def decode(self,data):
        return self.decoder.predict(data)
    def autoencode(self,data):
        return self.autoencoder.predict(data)
    def encode_binary(self,data):
        assert self.M == 2, "M={}, not 2".format(self.M)
        return self.encode(data).reshape(-1, self.N, self.M)[:,:,0].reshape(-1, self.N)
    def summary(self):
        self.autoencoder.summary()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import shlex, subprocess
    import os.path as p
    if p.exists("mnist_model/"):
        subprocess.call(shlex.split("rm -rf mnist_model/"))
    from mnist import mnist
    x_train, _, x_test, _ = mnist()
    ae = GumbelAE(784)
    ae.train(x_train,test_data=x_test)
    ae.save("mnist_model/")
    del ae
    ae = GumbelAE(784)
    ae.load("mnist_model/")

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
    plt.savefig('mnist_model/viz.png')



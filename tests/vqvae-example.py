# code from https://github.com/HenningBuhl/VQ-VAE_Keras_Implementation/blob/master/VQ_VAE_Keras_MNIST_Example.ipynb

import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import Input, Layer, Activation, Dense, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, SpatialDropout2D
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras import backend as K
from keras.utils import to_categorical
from keras.datasets import mnist, fashion_mnist
# Load data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data.
x_test = x_test / np.max(x_train)
x_train = x_train / np.max(x_train)

# Add input channel dimension.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Target dictionary.
target_dict = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


from latplan.util.layers import VQVAELayer


epochs = 1000 # MAX
batch_size = 64
validation_split = 0.1

# VQ-VAE Hyper Parameters.
embedding_dim = 32 # Length of embedding vectors.
num_embeddings = 128 # Number of embedding vectors (high value = high bottleneck capacity).
commitment_cost = 0.25 # Controls the weighting of the loss terms.

# EarlyStoppingCallback.
esc = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                                    patience=5, verbose=0, mode='auto',
                                    baseline=None, restore_best_weights=True)

# Encoder
input_img = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu')(input_img)
#x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
#x = Dropout(0.3)(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
#x = BatchNormalization()(x)
#x = Dropout(0.4)(x)

# VQVAELayer.
enc = Conv2D(embedding_dim, kernel_size=(1, 1), strides=(1, 1), name="pre_vqvae")(x)
enc_inputs = enc
x = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)

# Decoder.
x = Conv2DTranspose(64, (3, 3), activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = UpSampling2D()(x)
x = Conv2DTranspose(32, (3, 3), activation='relu')(x)
x = Conv2DTranspose(1, (3, 3))(x)

data_variance = np.var(x_train)
def loss(x, x_hat):
    return losses.mse(x, x_hat) / data_variance

# Autoencoder.
vqvae = Model(input_img, x)
vqvae.compile(loss=loss, optimizer='adam')
vqvae.summary()

history = vqvae.fit(x_train, x_train,
                    batch_size=batch_size, epochs=epochs,
                    validation_split=validation_split,
                    callbacks=[esc])

# Show original reconstruction.
n_rows = 5
n_cols = 8 # Must be divisible by 2.
samples_per_col = int(n_cols / 2)
sample_offset = np.random.randint(0, len(x_test) - n_rows * n_cols - 1)
#sample_offset = 0

img_idx = 0
plt.figure(figsize=(n_cols * 2, n_rows * 2))
for i in range(1, n_rows + 1):
    for j in range(1, n_cols + 1, 2):
        idx = n_cols * (i - 1) + j

        # Display original.
        ax = plt.subplot(n_rows, n_cols, idx)
        ax.title.set_text('({:d}) Label: {:s} ->'.format(
            img_idx,
            str(target_dict[np.argmax(y_test[img_idx + sample_offset])])))
        ax.imshow(x_test[img_idx + sample_offset].reshape(28, 28),
                  cmap='gray_r',
                  clim=(0, 1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction.
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.title.set_text('({:d}) Reconstruction'.format(img_idx))
        ax.imshow(vqvae.predict(
            x_test[img_idx + sample_offset].reshape(-1, 28, 28, 1)).reshape(28, 28),
            cmap='gray_r',
            clim=(0, 1))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        img_idx += 1
plt.show()

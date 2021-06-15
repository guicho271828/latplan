# global setting

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import numpy as np
np.set_printoptions(formatter={'float_kind':lambda x: "%.5f" % x})

# https://stackoverflow.com/questions/48979426/keras-model-accuracy-differs-after-loading-the-same-saved-model
from numpy.random import seed
seed(42) # keras seed fixing

import tensorflow as tf
tf.compat.v1.set_random_seed(42)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import keras.backend.tensorflow_backend as K
# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))

K.set_session(
    tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            device_count = {'CPU': 1, 'GPU': 1},
            gpu_options =
            tf.compat.v1.GPUOptions(
                per_process_gpu_memory_fraction=1.0,
                allow_growth=True,))))

from keras_radam      import RAdam
import keras.optimizers
setattr(keras.optimizers,"radam", RAdam)



from . import util
from . import puzzles
from . import model
from . import ama2model
from . import main


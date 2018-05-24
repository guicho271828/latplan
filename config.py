import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import keras.backend as K
tf.logging.set_verbosity(tf.logging.ERROR)
# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))

def load_session():
    K.set_session(
        tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=1,
                device_count = {'CPU': 1, 'GPU': 1},
                gpu_options =
                tf.GPUOptions(
                    per_process_gpu_memory_fraction=1.0,
                    allow_growth=True,))))

load_session()
clear_session = K.clear_session

def reload_session():
    clear_session()
    load_session()

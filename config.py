import tensorflow as tf
import keras.backend as K

# K.set_floatx('float16')
print("Default float: {}".format(K.floatx()))
K.set_session(
    tf.Session(
        config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options =
            tf.GPUOptions(
                per_process_gpu_memory_fraction=1.0,
                allow_growth=True,))))

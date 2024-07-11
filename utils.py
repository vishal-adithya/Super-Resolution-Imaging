import tensorflow as tf


def NORMALIZATION_FUNC(x):
    return x / 255.0 


class CALLBACK_TOOL():

    def MODEL_CHECKPOINT(path):
        x = tf.keras.callbacks.ModelCheckpoint(filepath=path)
        return x
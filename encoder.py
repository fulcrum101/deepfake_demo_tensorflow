import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """
    Builds a generic encoder layer made of Conv2D-IN-LeakyReLu.
    IN os optional, LeakyReLU may be replaced by relu.
    """
    conv = tf.keras.layers.Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization(axis=3)(x) # Batch Normalization per sample of data (contrast normalization)
    if activation == 'relu':
        x = tf.keras.layers.Activation('relu')(x)
    else:
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv(x)


    return x

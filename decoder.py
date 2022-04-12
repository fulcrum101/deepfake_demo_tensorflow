import tensorflow as tf
from tensorflow_addons.layers import InstanceNormalization

def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """
    Builds a generic decoder layer made of Conv2D-IN-LeakyReLU.
    IN is optional, LeakyReLU may be replaced by ReLU.
    :param inputs: (tensor) the decoder layer input.
    :param paired_inputs: (tensor) the encoder layer output provided by U-Net skip connection & concatenated to inputs.
    """
    conv = tf.keras.layers.Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')
    x = inputs
    if instance_norm:
        x = InstanceNormalization(axis=3)(x)
    if activation == 'relu':
        x = tf.keras.layers.Activation('relu')(x)
    else:
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = tf.keras.layers.concatenate([x, paired_inputs])

    return x
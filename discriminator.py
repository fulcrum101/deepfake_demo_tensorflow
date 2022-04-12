import tensorflow as tf
from encoder import encoder_layer

def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True,
                        name=None):
    """
    The discriminatior is a 4-layer encider that outputs either a 1-dim or a n x n-dim patch of probability that input is real.

    :param input_shape: (tuple) input shape
    :param kernel_size: (int) kernel size of decoder layers
    :param patchgan: (bool) wheteher the output is a patch or just 1-dim
    :param name: name assigned to discriminsor model
    :return: (string) name assigned to discriminator model
    """

    # Build a model with TensorFlow Functional API
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=1,
                      activation='leaky_relu',
                      instance_norm=False)

    if patchgan:
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        outputs = tf.keras.layers.Conv2D(1,
                                         kernel_size=kernel_size,
                                         strides=2,
                                         padding='same')(x)
    else:
        x = tf.keras.layers.Flaten()(x)
        x = Dense(1)(x)
        outputs = tf.keras.layers.Activations('linear')(x)

    discriminator = tf.keras.models.Model(inputs, outputs, name=name)

    return discriminator
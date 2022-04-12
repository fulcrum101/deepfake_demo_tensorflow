import tensorflow as tf
from encoder import encoder_layer
from decoder import decoder_layer

def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):
    """
    The generator is a U-Network made of a 4-layer encoder and 4-layer decoder.
    Layer n-1 is connected to layer i.
    :param input_shape: (tuple) input shape
    :param output_shape: (tuple) output shape
    :param kernel_size: (int) kernel size of encoder & decoder layers
    :param name: (string) name assigned to generated model
    :return: (Model) generator
    """

    # Build TensorFlow model with Functional API

    # Inputs
    inputs = tf.keras.layers.Input(shape=input_shape)
    channels = int(output_shape[-1])

    # Encoder layers
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       activation='leaky_relu',
                       strides=1)
    e2 = encoder_layer(e1,
                       64,
                       kernel_size=kernel_size,
                       activation='leaky_relu')
    e3 = encoder_layer(e2,
                       128,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e4 = encoder_layer(e3,
                       256,
                       activation='leaky_relu',
                       kernel_size=kernel_size)

    # Decoder layers
    d1 = decoder_layer(e4,
                       e3,
                       128,
                       kernel_size=kernel_size)
    d2 = decoder_layer(d1,
                       e2,
                       64,
                       kernel_size=kernel_size)
    d3 = decoder_layer(d2,
                       e1,
                       32,
                       kernel_size=kernel_size)

    # Outputs
    outputs = tf.keras.layers.Conv2DTranspose(channels,
                                              kernel_size=kernel_size,
                                              strides=1,
                                              activation='sigmoid',
                                              padding='same')(d3)
    generator = tf.keras.models.Model(inputs, outputs, name=name)

    return generator

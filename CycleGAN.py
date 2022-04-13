import datetime

import tensorflow as tf
from generator import build_generator
from discriminator import build_discriminator
import numpy as np

def build_cyclegan(shapes,
                   source_name='source',
                   target_name='target',
                   kernel_size=3,
                   patchgan=False,
                   identity=False):
    """
    Build the CycleGAN.
    1) Build target and source discriminators.
    2) Build target and source generators.
    3) Build the adversarial network.

    :param shapes: (tuple) source and target shapes
    :param source_name: (string) string to be appended on dis/gen models
    :param target_name: (string) string to be appended on dis/gen models
    :param kernel_size: (int) kernel size for the encoder/decoder or dis/gen models.
    :param patchgan: (whether to use patchgan on discriminators)
    :param identity: (bool) whether to use identity loss
    :return: (list) 2 generator, 2 discriminator, and 1 adversarial models
    """
    source_shape, target_shape = shapes
    lr = 2e-4
    decay = 6e-8
    gt_name = 'gen_' + target_name
    gs_name = 'gen_' + source_name
    dt_name = 'dis_' + target_name
    ds_name = 'dis_' + source_name

    # Build target and source generators
    g_target = build_generator(source_shape,
                               target_shape,
                               kernel_size=kernel_size,
                               name=gt_name)
    g_source = build_generator(target_shape,
                               source_shape,
                               kernel_size=kernel_size,
                               name=gs_name)
    print('---- TARGET GENERATOR ----')
    g_target.summary()
    print('---- SOURCE GENERATOR ----')
    g_source.summary()

    # Build target and source discriminators
    d_target = build_discriminator(target_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=dt_name)
    d_source = build_discriminator(source_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=ds_name)
    print('---- TARGET DISCRIMINATOR ----')
    d_target.summary()
    print('---- SOURCE DISCRIMINATOR ----')
    d_source.summary()

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr, decay=decay)
    d_target.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_source.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_target.trainable = False
    d_source.trainable = False

    # Build the computational graph for the adversarial model

    # Forward cycle network and target discriminator
    source_input = tf.keras.layers.Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    # Backward cycle network and source discriminator
    target_input = tf.keras.layers.Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    if identity:
        iden_souce = g_source(source_input)
        iden_target = g_target(target_input)
        loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10, 0.5, 0.5]
        inputs = [source_input, target_input]
        outptus = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   iden_souce,
                   iden_target]
    else:
        loss = ['mse', 'mse', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10.]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target]

        # Build adversarial model
        adv = tf.keras.models.Model(inputs, outputs, name='adversarial')
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr*0.5, decay=decay)
        adv.compile(loss=loss,
                    loss_weights=loss_weights,
                    optimizer=optimizer,
                    metrics=['accuracy'])
        print('---- ADVERSARIAL NETWORK ----')
        adv.summary()

        return g_source, g_target, d_source, d_target, adv

def train_cyclegan(models,
                       data,
                       params,
                       test_params,
                       test_generator):
    """
    Trains the CycleGAN.
    1) Train the target discriminator
    2) Train the source discriminator
    3) Train the forward and backward cycles of adversarial networks

    :param models: (Models) Source/Target Discriminator/Generator, Adversarial Model
    :param data: (tuple) source and target training data
    :param params: (tuple) network parameters
    :param test_params: (tuple) test parameters
    :param test_generator: (function) used for generating predicted target and source
    """
    # The models
    g_source, g_target, d_source, d_target, adv = models
    # Network parameters
    batch_size, train_steps, patch, model_name = params
    # Train dataset
    source_data, target_data, test_source_data, test_target_data = data
    titles, dirs = test_params

    # The generator image is saved 2000 steps
    save_interval = 2000
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    # Whether to use patchgan or not
    if patch > 1:
        d_patch = (patch, patch, 1)
        valid = np.ones((batch_size, ) + d_patch)
        fake = np.zeros((batch_size, ) + d_patch)
    else:
        valid = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

    valid_fake = np.concatenate((valid, fake))
    start_time = datetime.datetime.now()

    for steps in range(train_steps):
        # Sample a batch of real target data
        rand_indexes = np.random.randint(0,
                                             target_size,
                                             size=batch_size)
        real_target = target_data[rand_indexes]

        # Sample a batch of real source sata
        rand_indexes = np.random.randint(0,
                                             source_size,
                                             size=batch_size)
        real_source = source_data[rand_indexes]

        # Generate a batch of fake target data for real source data
        fake_target = g_target.predict(real_source)

        # Combine real and fake into one batch
        x = np.concatenate((real_target, fake_target))
        # Train the target discriminator using fake/real data
        metrics = d_target.train_on_batch(x, valid_fake)
        log = '%d: [d_target loss: %f]' % (steps, metrics[0])

        # Generate a batch of fake source data
        fake_source  = g_source.predict(real_target)
        x = np.concatenate((real_source, fake_source))
        # Train the source discriminator using fake/real data
        metrics = d_target.train_on_batch(x, valid_fake)
        log = '%d: [d_target loss: %f]' % (steps, metrics[0])

        # Generate a batch of fake sources data for real target data
        fake_source = g_source.predict(real_target)
        x = np.concatenate((real_source, fake_source))
        # Train the source discriminator using fake/real data
        metrics = d_source.train_on_batch(x, valid_fake)
        log = '%s [d_source loss: %f]' % (log, metrics[0])

        # Train the adversarial network using forward and backward cycles
        # The generated fake source and target data attempts to trick the discriminators
        x = [real_source, real_target]
        y = [valid, valid, real_source, real_target]
        metrics = adv.train_on_batch(x, y)
        elapsed_time = datetime.datetime.now() - start_time
        fmt = '%s [adv loss: %f] [time: %s]'
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)
        if (steps + 1) % save_interval == 0:
            test_generator((g_source, g_target),
                               (test_source_data, test_target_data),
                               step=steps+1,
                               titles=titles,
                               dirs=dirs,
                               show=False)

    # Save the models after training the generators
    g_source.save(model_name + '-g_source.h5')
    g_target.save(model_name + '-g_target.h5')





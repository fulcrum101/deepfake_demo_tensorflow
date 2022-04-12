from helper_functions import test_generator, load_data_cifar
from CycleGAN import build_cyclegan, train_cyclegan
def graycifar10_cross_colorcifar10(g_models=None):
    """
    Build and train a CycleGAN that can do
        grayscale <--> color cifar10 images
    """

    model_name = 'cyclegan_cifar10'
    batch_size = 32
    train_steps = 100000
    patchgan = True
    kernel_size = 3
    postfix = ('%dp' % kernel_size) if patchgan else ('%d' % kernel_size)

    data, shapes = load_data_cifar()
    source_data, _, test_source_data, test_target_data = data
    titles = ('CIFAR10 predicted source images.',
              'CIFAR10 predicted target images.',
              'CIFAR10 reconstructed source images.',
              'CIFAR10 reconstructed target images.')
    dirs = ('cifar10_source-%s' % postfix, 'cifar10_target-%s' % postfix)

    # generate predicted target(color) and source(gray) images
    if g_models is not None:
        g_source, g_target = g_models
        test_generator((g_source, g_target),
                                   (test_source_data, \
                                    test_target_data),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   show=True)
        return

    # build the cyclegan for cifar10 colorization
    models = build_cyclegan(shapes,
                            "gray-%s" % postfix,
                            "color-%s" % postfix,
                            kernel_size=kernel_size,
                            patchgan=patchgan)
    # patch size is divided by 2^n since we downscaled the input
    # in the discriminator by 2^n (ie. we use strides=2 n times)
    patch = int(source_data.shape[1] / 2 ** 4) if patchgan else 1
    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)
    # train the cyclegan
    train_cyclegan(models,
                   data,
                   params,
                   test_params,
                   test_generator)
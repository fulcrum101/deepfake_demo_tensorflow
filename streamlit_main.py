import streamlit as st
from PIL import Image
import tensorflow as tf

from tensorflow_addons.layers import InstanceNormalization


def predict_target(model_f, filename):
    model = tf.keras.models.load_model(model_f, custom_objects={'InstanceNormalization': InstanceNormalization})
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=1)
    im = tf.cast(tf.image.resize(img, size=[32, 32]), dtype=tf.float32)
    return tf.cast(tf.image.resize(img, size=[32, 32]), dtype=tf.uint16), model.predict(tf.expand_dims(im, axis=0))


def main():
    st.set_page_config(
        page_title="Deepfake demo",
        page_icon="🧊"
    )

    st.title('Deepfake demo')
    st.caption('Created using TensorFlow')
    st.write('This is demonstration of the power of the CycleGAN AKA deepfake technology.')
    st.write('NN trained on CIFAR10 dataset provided by [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/cifar10?authuser=3).')
    st.write('Used dataset includes 60000 32x32 colour images in 10 classes, with 6000 images per class.')
    st.write('There are 50000 training images and 10000 test images.')
    st.subheader('Training process')
    step_num = st.slider('Step number', 2000, 100000, step=2000)
    original_source = Image.open('cifar10_source-3p/reconstructed_source.png')
    original_target = Image.open('cifar10_target-3p/reconstructed_target.png')

    step_source = Image.open(f'cifar10_source-3p/{str(step_num).zfill(6)}.png')
    step_target = Image.open(f'cifar10_target-3p/{str(step_num).zfill(6)}.png')
    col1, col2 = st.columns(2)
    col1.caption('Original sources and targets')
    col2.caption('Sources and targets recreated by AI')
    col1.image(original_source)
    col2.image(step_source)
    col1.image(original_target)
    col2.image(step_target)
    st.subheader('Deepfakes on portraits')
    col1, col2 = st.columns(2)
    col1.caption('Original portrait')
    col2.caption('AI coloured portrait')
    t_or, t_col = predict_target('cyclegan_cifar10-g_target.h5', 'portraits/alan_turing.png')
    col1.image(t_or)
    col2.image(t_col)
    t_or, t_col = predict_target('cyclegan_cifar10-g_target.h5', 'portraits/albert_einstein.png')
    col1.image(t_or)
    col2.image(t_col)
    t_or, t_col = predict_target('cyclegan_cifar10-g_target.h5', 'portraits/nikola_tesla.png')
    col1.image(t_or)
    col2.image(t_col)


if __name__ == '__main__':
    main()

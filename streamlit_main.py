import streamlit as st
from PIL import Image
import tensorflow as tf

from tensorflow_addons.layers import InstanceNormalization


def predict_target(model, filename):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=1)
    im = tf.cast(tf.image.resize(img, size=[32, 32]), dtype=tf.float32)
    return tf.squeeze(model.predict(tf.expand_dims(im, axis=0)))


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

    model = tf.keras.models.load_model('cyclegan_cifar10-g_target.h5', custom_objects={'InstanceNormalization': InstanceNormalization})
    st.subheader('Deepfake on portraits')
    col1, col2, col3 = st.columns(3)
    turing = predict_target(model, 'portraits/alan_turing.png')
    col1.image(turing, use_column_width='always', caption='Alan Turing')
    einstein = predict_target(model, 'portraits/albert_einstein.png')
    col2.image(einstein, use_column_width='always', caption='Albert Einstein')
    tesla = predict_target(model, 'portraits/nikola_tesla.png')
    col3.image(tesla, use_column_width='always', caption='Nikola Tesla')

    st.subheader('Deepfake on custom images')
    col1, col2, col3 = st.columns(3)
    img1 = predict_target(model, 'custom_images/1.png')
    col1.image(img1, use_column_width='always', caption='Tree')
    img2 = predict_target(model, 'custom_images/2.png')
    col2.image(img2, use_column_width='always', caption='Starry sky')
    img3 = predict_target(model, 'custom_images/3.png')
    col3.image(img3, use_column_width='always', caption='Wave')




if __name__ == '__main__':
    main()

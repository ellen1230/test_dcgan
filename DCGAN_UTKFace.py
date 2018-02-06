import keras as k
import numpy as np
from keras import optimizers

from keras.layers import Input, Dense, BatchNormalization, initializers, Merge, Concatenate
from keras.layers.core import Activation, Reshape, Flatten, RepeatVector, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

import tensorflow as tf

from PIL import Image

def generator_model():
    kernel_initializer = initializers.random_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    inputs_z = Input(shape=(100, ))
    # inputs_y = Input(shape=(10, ))
    # input_y_conv = Input(shape=(1, 1, 10))

    current = Dense(64*8*4*4, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs_z)
    current = Reshape(target_shape=(4, 4, 64 * 8))(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(4, 4, 64 * 8),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Conv2DTranspose(filters=64*4, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(8, 8, 64*4),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Conv2DTranspose(filters=64*2, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(16, 16, 64*2),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Conv2DTranspose(filters=64*1, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(32, 32, 64*1),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Conv2DTranspose(filters=3, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Activation('tanh')(current)

    return Model(inputs=inputs_z, outputs=current)

def discriminator_model():
    kernel_initializer = initializers.truncated_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    inputs_img = Input(shape=(64, 64, 3))

    current = Conv2D(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs_img)
    current = Lambda(lrelu, output_shape=(32, 32, int(current.shape[3])))(current)

    current = Conv2D(filters=64 * 2, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(16, 16, int(current.shape[3])),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Lambda(lrelu, output_shape=(16, 16, int(current.shape[3])))(current)

    current = Conv2D(filters=64 * 4, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(8, 8, int(current.shape[3])),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Lambda(lrelu, output_shape=(8, 8, int(current.shape[3])))(current)

    current = Conv2D(filters=64 * 8, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(4, 4, int(current.shape[3])),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Lambda(lrelu, output_shape=(4, 4, int(current.shape[3])))(current)

    kernel_initializer = initializers.random_normal(stddev=0.02)
    current = Reshape(target_shape=(4 * 4 * 512, ))(current)
    current = Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Activation('sigmoid')(current)

    return Model(inputs=inputs_img, outputs=current)

def dcganModel(generator, discriminator):
    inputs_z = Input(shape=(100,))
    x = generator(inputs_z)
    discriminator.trainable = False
    dcganOutput = discriminator(x)
    return Model(inputs=inputs_z, outputs=dcganOutput)

def load_mnist():
    import os
    from glob import glob
    # import scipy
    from PIL import Image

    file_names = glob(os.path.join('./data/UTKFace/', '*.jpg'))
    x_train = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        # image = scipy.misc.imread(path).astype(np.float)
        image = np.array(Image.open(file_name).resize((64, 64)))
        image = image.astype(np.float32)/ 127.5 - 1
        x_train.append(image.tolist())
    return np.array(x_train)

def train(batchsize, numepoch):
    data_x = load_mnist()
    numbatch = int(data_x.shape[0]/batchsize)

    # Model
    generator = generator_model()
    discriminator = discriminator_model()
    dcgan = dcganModel(generator, discriminator)

    # Optimizers
    # sgd = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    d_adam = Adam(lr=0.0002, beta_1=0.5)
    g_adam = Adam(lr=0.0002, beta_1=0.5)

    # Compile
    # generator.compile(optimizer=adam, loss="binary_crossentropy")
    dcgan.compile(optimizer=g_adam, loss="binary_crossentropy")
    discriminator.trainable = True
    discriminator.compile(optimizer=d_adam, loss="binary_crossentropy")

    d_loss = []
    g_loss1 = []
    g_loss2 = []

    # fit
    for epoch in range(numepoch):
        print("Epoch: ", epoch)

        for i in range(numbatch):
            # real image+label
            real_image = data_x[i * batchsize:(i + 1) * batchsize, :]
            real_label = np.ones(batchsize)

            # noise image+label
            noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            noise_image = generator.predict(noise, verbose=0)
            noise_label = np.zeros(batchsize)

            img_train_batch = np.concatenate((noise_image, real_image), axis=0)
            label_train_batch = np.concatenate((noise_label, real_label), axis=0)
            d_loss.append(discriminator.train_on_batch(img_train_batch, label_train_batch))

            # noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            discriminator.trainable = False
            g_loss1.append(dcgan.train_on_batch(noise, np.ones(batchsize)))
            g_loss2.append(dcgan.train_on_batch(noise, np.ones(batchsize)))
            discriminator.trainable = True

            if (epoch % 20 == 0) or (epoch == 1):
                save_image(noise_image, i, epoch)
                generator.save(filepath="weight/g_e"+str(epoch) + 'b' + str(i) + ".h5", overwrite=True)
                print("After epoch: ", epoch)
                print("discriminator loss: ", d_loss[-1], "\t g loss1: ", g_loss1[-1], "\t g loss2: ", g_loss2[-1])

    np.save('metric/dLoss.npy', np.array(d_loss))
    np.save('metric/gLoss1.npy', np.array(g_loss1))
    np.save('metric/gLoss2.npy', np.array(g_loss2))

def save_image(images, numbatch, epoch):
    from scipy.misc import imsave
    num_images = images.shape[0]
    num_picture = int(np.sqrt(num_images))
    picture = np.zeros((64 * num_picture, 64 * num_picture, 3))

    for i in range(num_picture):
        for j in range(num_picture):
            index = i * num_picture + j
            image = images[index]
            picture[i * 64:(i + 1) * 64, j * 64:(j + 1) * 64, :] = image

    picture = ((picture + 1)/2 * 255.0).astype(np.uint8)
    path = "generate/e" + str(epoch) + 'b' + str(numbatch) + ".png"
    imsave(path, picture)

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def init_param(shape):
    return

if __name__ == '__main__':
    train(50, 200)










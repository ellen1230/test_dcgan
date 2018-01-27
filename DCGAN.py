import keras as k
import numpy as np
from keras import optimizers

from keras.layers import Input, Dense, BatchNormalization, initializers
from keras.layers.core import Activation, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Convolution2D

from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

def generator_model():
    inputs = Input(shape=(100, ))
    # kernel_initializer = initializers.random_normal(stddev=0.02)

    dense1 = Dense(1024, activation="relu")(inputs)
    dense2 = Dense(128*7*7)(dense1)
    # bn = BatchNormalization()(dense2)
    bn_a = Activation(activation="relu")(dense2)
    bn_r = Reshape(target_shape=(7, 7, 128))(bn_a)
    up1 = UpSampling2D(size=(2, 2))(bn_r)
    #conv1 = Conv2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(up1)
    conv1 = Convolution2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(up1)
    up2 = UpSampling2D(size=(2, 2))(conv1)
    #conv2 = Conv2D(filters=1, kernel_size= (5, 5), padding="same", activation="tanh")(up2)
    conv2 = Convolution2D(filters=1, kernel_size=(5, 5), padding="same", activation="tanh")(up2)

    return Model(inputs=inputs, outputs=conv2)

def discriminator_model():
    inputs = Input(shape=(28*28, ))

    x = Reshape((28, 28, 1))(inputs)
    # conv1 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")(x)
    # conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Convolution2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(x)
    max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(filters=128, kernel_size=(5, 5), padding="same")(max1)
    # conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Convolution2D(filters=128, kernel_size=(5, 5), padding="same")(max1)
    max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    flat = Flatten()(max2)
    dense1 = Dense(units=1024, activation="relu")(flat)
    # dense1 = LeakyReLU(alpha=0.2)(dense1)
    dense2 = Dense(units=1, activation="sigmoid")(dense1)

    return Model(inputs=inputs, outputs=dense2)

def dcganModel(generator, discriminator):
    dcganInput = Input(shape=(100, ))
    x = generator(dcganInput)
    discriminator.trainable = False
    dcganOutput = discriminator(x)
    return Model(inputs=dcganInput, outputs=dcganOutput)

def train(batchsize, numepoch):
    # mnist = input_data.read_data_sets("data/", one_hot=True)
    # (x_train, y_train) = (mnist.train.images, mnist.train.labels)
    # (x_test, y_test) = (mnist.test.images, mnist.test.labels)
    # x_train = (x_train.astype(np.float32) - 0.5) / 0.5

    import os
    from glob import glob
    from PIL import Image

    file_names = glob(os.path.join('./data/UTKFace/', '*.jpg'))
    x_train = []
    for i in range(len(file_names)):
        file_name = file_names[i]
        image = np.array(Image.open(file_name).convert('L').resize((28, 28))).flatten()
        image = (image.astype(np.float32) - 127.5) / 127.5
        x_train.append(image.tolist())
    x_train = np.array(x_train)

    numbatch = int(x_train.shape[0]/batchsize)

    # Model
    discriminator = discriminator_model()
    generator = generator_model()
    dcgan = dcganModel(generator, discriminator)

    # Optimizers
    # sgd = optimizers.SGD(lr=0.0005, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.002, beta_1=0.5)

    # Compile
    generator.compile(optimizer=adam, loss="binary_crossentropy")
    dcgan.compile(optimizer=adam, loss="binary_crossentropy")
    discriminator.trainable = True
    discriminator.compile(optimizer=adam, loss="binary_crossentropy")

    # fit
    for epoch in range(numepoch):
        print("Epoch: ", epoch)

        for i in range(numbatch):
            noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            # noise image+label
            noise_image, noise_label = generator.predict(noise, verbose=0), np.zeros(batchsize)
            height, length = noise_image[0].shape[0], noise_image[0].shape[1]
            noise_image = noise_image.reshape(-1, batchsize, height*length)[0]
            # real image+label
            # image_batch = mnist.train.next_batch(batchsize)
            # real_image, real_label = image_batch[0], np.ones(batchsize)
            real_image, real_label = x_train[i*batchsize:(i+1)*batchsize,:], np.ones(batchsize)
            real_image = (real_image.astype(np.float32) - 0.5) / 0.5

            x_train_batch = np.concatenate((noise_image, real_image), axis=0)
            y_train_batch = np.concatenate((noise_label, real_label), axis=0)
            d_loss = discriminator.train_on_batch(x_train_batch, y_train_batch)

            noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            discriminator.trainable = False
            dcgan_loss = dcgan.train_on_batch(noise, np.ones(batchsize))
            discriminator.trainable = True

            if (epoch % 20 == 0) or (epoch == 1):
                save_image(noise_image, i, epoch)
                generator.save(filepath="weight/g_e"+str(epoch)+ 'b' + str(i) + ".h5", overwrite=True)
                print("After epoch: ", epoch)
                print("dcgan Loss: ", dcgan_loss, "\t discriminator loss", d_loss)



def save_image(images, numbatch, epoch):
    num_images = images.shape[0]
    num_picture = int(np.sqrt(num_images))
    picture = np.zeros([28*num_picture, 28*num_picture])

    for i in range(num_picture):
        for j in range(num_picture):
            index = i * num_picture + j
            image = images[index].reshape(28, 28)
            picture[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = image

    # image = images[0].reshape(28, 28)
    # for i in range(1, images.shape[0]):
    #     image = np.append(image, images[i].reshape(28, 28), axis=1)
    picture= picture * 127.5 + 127.5
    picture = Image.fromarray(picture, mode="L")
    picture.save("generate/e" + str(epoch) + 'b' + str(numbatch) + ".jpg")

    # i = 0
    # for image in images:
    #     image = image * 127.5 + 127.5
    #     image = Image.fromarray(image.reshape(28, 28), mode="L")
    #
    #     image.save("generate/e" + str(epoch) + "_b" + str(numbatch)+"_"+str(i) + ".jpg")
    #     i = i + 1

def init_param(shape):
    return


if __name__ == '__main__':
    train(50, 200)










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

    inputs_img = Input(shape=(100, ))
    inputs_y = Input(shape=(10, ))
    input_y_conv = Input(shape=(1, 1, 10))

    # current = tf.concat([current, inputs_y], axis=-1)
    # current = Merge(mode='concat', concat_axis=-1)([inputs_img, inputs_y])
    current = Concatenate(axis=-1)([inputs_img, inputs_y])
    current = Dense(1024, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization,output_shape=(int(current.shape[1]),),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    # current = tf.layers.batch_normalization(inputs=current)
    current = Activation(activation='relu')(current)

    current = Concatenate(axis=-1)([current, inputs_y])
    current = Dense(128*7*7, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(int(current.shape[1]),),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Reshape(target_shape=(7, 7, 128))(current)
    inputs_y_repeat = Lambda(concatenate, output_shape=(7, 7, 10), arguments={'times': 7})(input_y_conv)
    current = Concatenate(axis=-1)([current, inputs_y_repeat])

    current = Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(14, 14, int(current.shape[3])),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Activation(activation='relu')(current)

    current = Conv2DTranspose(filters=1, kernel_size=(5, 5), padding='same', strides=(2, 2),
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Activation('sigmoid')(current)

    return Model(inputs=[inputs_img, inputs_y, input_y_conv], outputs=current)

def discriminator_model():
    kernel_initializer = initializers.truncated_normal(stddev=0.02)
    bias_initializer = initializers.constant(value=0.0)

    inputs_img = Input(shape=(28, 28, 1))
    inputs_y = Input(shape=(10,))
    input_y_conv = Input(shape=(1, 1, 10))

    # current = Reshape((28, 28, 1))(inputs_img)
    inputs_y_repeat = Lambda(concatenate, output_shape=(28, 28, 10), arguments={'times': 28})(input_y_conv)
    current = Concatenate(axis=-1)([inputs_img, inputs_y_repeat])

    current = Conv2D(filters=1+10, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(lrelu, output_shape=(14, 14, int(current.shape[3])))(current)

    inputs_y_repeat = Lambda(concatenate, output_shape=(14, 14, 10), arguments={'times': 14})(input_y_conv)
    current = Concatenate(axis=-1)([current, inputs_y_repeat])

    current = Conv2D(filters=64+10, kernel_size=(5, 5), padding='same', strides=(2, 2),
                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(7, 7, int(current.shape[3])),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Lambda(lrelu, output_shape=(7, 7, int(current.shape[3])))(current)

    kernel_initializer = initializers.random_normal(stddev=0.02)

    current = Reshape(target_shape=(7*7*74,))(current)
    current = Concatenate(axis=-1)([current, inputs_y])
    current = Dense(1024, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Lambda(tf.layers.batch_normalization, output_shape=(int(current.shape[1]),),
                     arguments={'momentum': 0.9, 'epsilon': 1e-5, 'scale': True})(current)
    current = Lambda(lrelu, output_shape=(int(current.shape[1]), ))(current)

    current = Concatenate(axis=-1)([current, inputs_y])
    current = Dense(1, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(current)
    current = Activation('sigmoid')(current)
    # conv1 = Conv2D(filters=64, kernel_size=(5, 5), padding="same")(x)
    # conv1 = LeakyReLU(alpha=0.2)(conv1)
    # Convolution2D is another name of Conv2D
    # conv1 = Convolution2D(filters=64, kernel_size=(5, 5), padding="same", activation="relu")(x)
    # max1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(filters=128, kernel_size=(5, 5), padding="same")(max1)
    # conv2 = LeakyReLU(alpha=0.2)(conv2)
    # conv2 = Convolution2D(filters=128, kernel_size=(5, 5), padding="same")(max1)
    # max2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # flat = Flatten()(max2)
    # dense1 = Dense(units=1024, activation="relu")(flat)
    # # dense1 = LeakyReLU(alpha=0.2)(dense1)
    # dense2 = Dense(units=1, activation="sigmoid")(dense1)

    return Model(inputs=[inputs_img, inputs_y, input_y_conv], outputs=current)

def dcganModel(generator, discriminator):
    inputs_img = Input(shape=(100,))
    inputs_y = Input(shape=(10,))
    input_y_conv = Input(shape=(1, 1, 10))
    x = generator([inputs_img, inputs_y, input_y_conv])
    discriminator.trainable = False
    dcganOutput = discriminator([x, inputs_y, input_y_conv])
    return Model(inputs=[inputs_img, inputs_y, input_y_conv], outputs=dcganOutput)

def load_mnist():
    import os
    data_dir = os.path.join("./data", 'mnist')

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.0

    return X / 255., y_vec

def train(batchsize, numepoch):
    data_x, data_y = load_mnist()
    data_y_conv = np.reshape(data_y, [len(data_y), 1, 1, 10])
    # x_train = (x_train.astype(np.float32) - 0.5) / 0.5

    # import os
    # from glob import glob
    # from PIL import Image
    #
    # file_names = glob(os.path.join('./data/UTKFace/', '*.jpg'))
    # x_train = []
    # for i in range(len(file_names)):
    #     file_name = file_names[i]
    #     image = np.array(Image.open(file_name).convert('L').resize((28, 28))).flatten()
    #     image = (image.astype(np.float32) - 127.5) / 127.5
    #     x_train.append(image.tolist())
    # x_train = np.array(x_train)

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
            real_y = data_y[i * batchsize:(i + 1) * batchsize, :]
            real_y_conv = data_y_conv[i * batchsize:(i + 1) * batchsize, :]
            real_label = np.ones(batchsize)

            # noise image+label
            noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            noise_image = generator.predict([noise, real_y, real_y_conv], verbose=0)
            noise_label = np.zeros(batchsize)
            # height, length = noise_image[0].shape[0], noise_image[0].shape[1]
            # noise_image = noise_image.reshape(-1, batchsize, height*length)[0]
            # real_image = (real_image.astype(np.float32) - 0.5) / 0.5

            img_train_batch = np.concatenate((noise_image, real_image), axis=0)
            y_train_batch = np.concatenate((real_y, real_y), axis=0)
            y_conv_batch = np.concatenate((real_y_conv, real_y_conv), axis=0)
            label_train_batch = np.concatenate((noise_label, real_label), axis=0)
            d_loss.append(discriminator.train_on_batch([img_train_batch, y_train_batch, y_conv_batch], label_train_batch))

            # noise = np.random.uniform(-1, 1, size=(batchsize, 100))
            discriminator.trainable = False
            g_loss1.append(dcgan.train_on_batch([noise, real_y, real_y_conv], np.ones(batchsize)))
            g_loss2.append(dcgan.train_on_batch([noise, real_y, real_y_conv], np.ones(batchsize)))
            discriminator.trainable = True

            if (epoch % 20 == 0) or (epoch == 1):
                save_image(noise_image, i, epoch)
                generator.save(filepath="weight/g_e"+str(epoch)+ 'b' + str(i) + ".h5", overwrite=True)
                print("After epoch: ", epoch)
                print("discriminator loss: ", d_loss[-1], "\t g loss1: ", g_loss1[-1], "\t g loss2: ", g_loss2[-1])

    np.save('metric/dLoss.npy', np.array(d_loss))
    np.save('metric/gLoss1.npy', np.array(g_loss1))
    np.save('metric/gLoss2.npy', np.array(g_loss2))

def save_image(images, numbatch, epoch):
    from scipy.misc import imsave
    num_images = images.shape[0]
    num_picture = int(np.sqrt(num_images))
    picture = np.zeros((28*num_picture, 28*num_picture))

    for i in range(num_picture):
        for j in range(num_picture):
            index = i * num_picture + j
            image = images[index][:, :, 0]
            picture[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = image

    picture = (picture * 255.0).astype(np.uint8) # gray
    # picture = picture * 255
    path = "generate/e" + str(epoch) + 'b' + str(numbatch) + ".png"
    imsave(path, picture)
    # picture.save("generate/e" + str(epoch) + 'b' + str(numbatch) + ".png")

    # manifold_h = int(np.floor(np.sqrt(num_images)))
    # manifold_w = int(np.ceil(np.sqrt(num_images)))



    # i = 0
    # for image in images:
    #     image = image * 127.5 + 127.5
    #     image = Image.fromarray(image.reshape(28, 28), mode="L")
    #
    #     image.save("generate/e" + str(epoch) + "_b" + str(numbatch)+"_"+str(i) + ".jpg")
    #     i = i + 1

# def conv_cond_concat(x, y):
#   """Concatenate conditioning vector on feature map axis."""
#   x_shapes = x.get_shape()
#   y_shapes = y.get_shape()
#   return concat([
#     x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)
def concat(x, y):
    return tf.concat([x, y], axis=-1)

def concatenate(x, times):
    list = []
    for i in range(times):
        list.append(x)
    x = Concatenate(axis=1)(list)
    list = []
    for i in range(times):
        list.append(x)
    output = Concatenate(axis=2)(list)
    return output

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def init_param(shape):
    return


if __name__ == '__main__':
    train(50, 200)










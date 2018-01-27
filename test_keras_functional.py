import keras as k

from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist


from tensorflow.examples.tutorials.mnist import input_data

inputs = Input(shape=(784,))

x = Dense(units=100, activation="relu")(inputs)
x = Dense(units=50, activation="relu")(x)
predict = Dense(units=10, activation="softmax")(x)

model = Model(inputs=inputs, outputs=predict)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

#----------------model可视化
# from keras.utils import plot_model
# plot_model(model, to_file= "1.jpg")

train = input_data.read_data_sets("data/", one_hot=True).train
history = model.fit(train.images, train.labels, batch_size=50, epochs=1)
#
# #---------------引用训练好的model
# from keras.layers import TimeDistributed
# input_seq = Input(shape=(20, 784))
# Timemodel = TimeDistributed(model)(input_seq)

#----------------两个输入公用一个网络
from keras.layers import LSTM
a = Input(shape=(5, 3))
b = Input(shape=(5, 3))

lstm = LSTM(10)
encode_a = lstm(a)
encode_b = lstm(b)

assert lstm.get_output_at(0) == encode_a
assert lstm.get_output_at(1) == encode_b
print(1)
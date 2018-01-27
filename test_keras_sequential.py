import numpy as np
import keras as k

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD

# data
x_train = np.random.random((1000, 20))
y_train = k.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = k.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# model
model = Sequential()
model.add(Dense(64, input_dim=20, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(10, activation="softmax"))

sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=50, epochs=2)
score = model.evaluate(x_test, y_test, batch_size=50)
print(score)

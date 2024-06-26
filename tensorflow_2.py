#!/usr/bin/env python
from datetime import datetime

import tensorflow as tf

from tensorflow.keras import datasets, layers, models, callbacks

# Using provided cifar10 data from keras dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalizing pixel values to be between 0, 1
x_train, x_test = x_train / 255.0, x_test / 255.0

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10),
])

# Set model parameters
# Note: in tensorflow, explicitly set as fields to model.fit
optimizer = 'adam'
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
epochs = 10
batch_size = 128

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)


# "compiling" model just establishes some configuation options -- still eager execution
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# fit model
model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

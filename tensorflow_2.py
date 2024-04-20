import tensorflow as tf

from tensorflow.keras import datasets, layers, models

def prepare_model():
    """
    Prepare a standard 3-conv layer model.
    :return: the model
    """
    retval = models.Sequential()

    retval.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    retval.add(layers.MaxPooling2D((2, 2)))
    retval.add(layers.Conv2D(64, (3, 3), activation='relu'))
    retval.add(layers.MaxPooling2D((2, 2)))
    retval.add(layers.Conv2D(64, (3, 3), activation='relu'))
    retval.add(layers.Flatten())
    retval.add(layers.Dense(64, activation='relu'))
    retval.add(layers.Dense(10))

    return retval

# ***** LOAD DATA ***** #

# Using provided cifar10 data from keras dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalizing pixel values to be between 0, 1
x_train, x_test = x_train / 255.0, x_test / 255.0

model = prepare_model()
# Set model parameters
# Note: in tensorflow, explicitly set as fields to model.fit
optimizer='adam'
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
epochs = 10
batch_size = 128

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model.fit(x_train, y_train,
          epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras import datasets, layers, models

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def visualize(train_images, train_labels):
    """
    Visualize the first 25 images of the cifar image set.

    :param train_images: Images to visualize
    :param train_labels: Provided class labels for images
    """
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i])
        # The CIFAR labels happen to be arrays,
        # which is why you need the extra index
        plt.xlabel(class_names[train_labels[i][0]])
    plt.show()


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


(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
# Normalizing pixel values to be between 0, 1
x_train, x_test = x_train / 255.0, x_test / 255.0

visualize(x_train, y_train)
model = prepare_model()
model.summary()

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
              , metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_test, y_test))

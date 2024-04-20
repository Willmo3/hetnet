import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import cifar10

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


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalizing pixel values to be between 0, 1
x_train, x_test = x_train / 255.0, x_test / 255.0

visualize(x_train, y_train)

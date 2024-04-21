import optax

# Using pytorch data abstraction
import jax
import jax.numpy as jnp

# Do need some numpy-specific functions
import numpy as np
from flax import linen as nn
from flax.training import train_state

import torch
import torchvision
import torchvision.transforms as transforms


# ***** DATA LOADERS ***** #

def get_datasets():
    """
    Prepare Cifar10 dataset from tensorflow dataset loader.
    tfds seems to be the preferred dataset for flax users.
    However, it has infinite recursion on Python 3.12 (no, this isn't a joke.)
    I'm going back to torch.
    :return: The training and test data.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # built in loader datatype controls batch siz
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    return train_set, val_set


# ***** NETWORK ***** #

class CNN(nn.Module):
    """
    Flax convolutional neural network.
    nn.module superclass contains many useful methods like apply.
    We must override __call__ method.
    """
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


# ***** JAX SPECIFIC FNS ***** #

def compute_metrics(predictions, labels):
    """
    Helper to compute relevant metrics
    :param predictions: Final predictions
    :param labels: Actual labels of dataset
    :return: Tuple with loss and accuracy
    """
    loss = (jnp.mean
            (optax.softmax_cross_entropy(predictions, jax.nn.one_hot(labels, num_classes=10))))
    accuracy = jnp.mean(jnp.argmax(predictions, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy
    }

    return metrics


# jit annotation: apply XLA jit-compilation
# Useful for GPU operations
@jax.jit
def train_step(state, batch):
    """
    Train a single batch of data.
    NOTE: in Google-provided Flax examples, use a state machine pattern, so keeping state param.
    :param state: Current state in training; will be mutated
    :param batch: Batch to consider
    :return: newly modified state + performance metrics
    """
    def loss_fn(params):
        """
        Cross entropy loss.
        :param params:
        Input parameters to apply to model; model's initial weights are already initialized.
        :return: The loss and predictions
        """
        predictions = CNN().apply({'params': params}, batch['image'])
        loss = jnp.mean(optax.softmax_cross_entropy(
            logits=predictions,
            labels=jax.nn.one_hot(batch['label'], num_classes=10)))
        return loss, predictions

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, predictions), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(predictions, batch['label'])
    return state, metrics


@jax.jit
def eval_step(params, batch):
    """
    Single evaluation of model
    :param params: Parameters for CNN application, per flax API
    :param batch: Batch of images to work with
    :return: Metrics
    """
    predictions = CNN().apply({'params': params}, batch['image'])
    return compute_metrics(predictions, batch['label'])


def train_epoch(state, train_ds, batch_size, rng, epoch):
    """
    Run a single epoch
    :param state: Current state of model.
    :param train_ds: Training data to operate over
    :param batch_size: size of a single batch
    :param rng random number generator
    :param epoch: epoch we're on
    :return: The state of the model after a single epoch.
    """

    train_ds_size = len(train_ds.data)
    steps_per_epoch = train_ds_size // batch_size

    # Randomly select data points for each batch
    perms = jax.random.permutation(rng, len(train_ds.data))
    perms = perms[:steps_per_epoch * batch_size]  # Skip an incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    batch_metrics = []

    for perm in perms:
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch)
        batch_metrics.append(metrics)

    training_batch_metrics = jax.device_get(batch_metrics)
    training_epoch_metrics = {
        k: np.mean([metrics[k] for metrics in training_batch_metrics])
        for k in training_batch_metrics[0]}

    print('Training - epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, training_epoch_metrics['loss'], training_epoch_metrics['accuracy'] * 100))
    return state, training_epoch_metrics


def eval_model(model, test_ds):
    metrics = eval_step(model, test_ds)
    metrics = jax.device_get(metrics)
    eval_summary = jax.tree_map(lambda x: x.item(), metrics)
    return eval_summary['loss'], eval_summary['accuracy']


# ***** PREPARE MODEL FOR TRAINING ***** #

# Using PyTorch datasets because tfds is broken right now
train_ds, test_ds = get_datasets()
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)

model = CNN()
# Initialize model params (i.e. weights) randomly
params = model.init(init_rng, jnp.ones([1, 28, 28, 1]))['params']

# For now, just using defaults for optax adam
optimizer = optax.adam(learning_rate=0.001)
# flax uses state model.
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)


# ***** TRAIN MODEL ***** #
num_epochs = 10
batch_size = 128

for epoch in range(1, num_epochs + 1):
    # Use a separate PRNG key to permute image data during shuffling
    rng, input_rng = jax.random.split(rng)
    # Run an optimization step over a training batch
    state, train_metrics = train_epoch(state, train_ds, batch_size, epoch, input_rng)
    # Evaluate on the test set after each training epoch
    test_loss, test_accuracy = eval_model(state.params, test_ds)
    print('Testing - epoch: %d, loss: %.2f, accuracy: %.2f' % (epoch, test_loss, test_accuracy * 100))



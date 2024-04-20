#!/usr/bin/env python

import time

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn


# ***** TRAINING HELPERS ***** #

# Explicit training function from Dr. Nathan Sprague's optimizer
def train(dataloader, model, loss_fn, optimizer):
    """
    One training iteration.
    From Dr. Sprague's optimizer.

    :param dataloader: PyTorch dataloader obj to get data from
    :param model: model to train on
    :param loss_fn: Target loss fn (i.e. cross entropy)
    :param optimizer: optimization fn (i.e. Adam; momentum + regularization)
    :return: final loss
    """

    model.train()  # Set the "training" flag to true for the model.
    total_loss = 0

    start_time = time.time()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction error
        # Much more explicit
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch % 100 == 0:
            print(".", end="", flush=True)

    print(f"Epoch time: {time.time() - start_time:.4f}(s)")

    return total_loss / len(dataloader)


def test(dataloader, model, loss_fn):
    """
    Test model performance.
    From Dr. Sprague

    :param dataloader: Test set
    :param model: Trained model
    :param loss_fn: Loss function for
    :return: Loss in test instances + proportion correctly identiifed
    """

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss, correct


# ***** LOAD DATA ***** #

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 128

# built in loader datatype controls batch size

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

# ***** DEFINE MODEL ***** #

model = nn.Sequential(
    nn.Conv2d(3, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(64, 64, 3),
    nn.ReLU(),
    nn.Flatten(),
    # dense layer of size 64 over image reduced to size 8 * 8
    nn.Linear(4 * 4 * 64, 64),
    nn.ReLU(),
    # one output unit for each image
    nn.Linear(64, 10),
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10

for epoch in range(epochs):
    train(train_loader, model, loss_fn, optimizer)
    train_loss, train_acc = test(train_loader, model, loss_fn)
    val_loss, val_acc = test(val_loader, model, loss_fn)

    train_str = f"loss: {train_loss:.4f} accuracy: {train_acc:.3f} "
    val_str = f"validation loss: {val_loss:.4f} validation accuracy: {val_acc:.3f}"
    print(f"Epoch {epoch + 1} " + train_str + val_str)

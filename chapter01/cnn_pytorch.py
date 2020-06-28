"""Training a CNN (convolutional neural network) on CIFAR-10 using PyTorch
    Author: Roi Yehoshua
    Date: June 2020
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import time

import plot_utils

# Load the data
transform = transforms.ToTensor()
train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# Create a validation set
train_size = int(0.9 * len(train_set))
validation_size = int(0.1 * len(train_set))
train_set, validation_set = torch.utils.data.random_split(train_set,
                                                          [train_size, validation_size])

# Define the data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# Examine the shape of the input
# data_iter = iter(train_loader)
# x_train, y_train = data_iter.next()
# print('x_train shape:', x_train.shape)

# Define the model
model = nn.Sequential(               # input tensor has shape (3, 32, 32)
    nn.Conv2d(3, 32, 3, padding=1),  # (32, 32, 32)
    nn.ReLU(),
    nn.Conv2d(32, 32, 3),            # (32, 30, 30)
    nn.ReLU(),
    nn.MaxPool2d(2),                 # (32, 15, 15)
    nn.Dropout(0.25),

    nn.Conv2d(32, 64, 3, padding=1), # (64, 15, 15)
    nn.ReLU(),
    nn.Conv2d(64, 64, 3),            # (64, 13, 13)
    nn.ReLU(),
    nn.MaxPool2d(2),                 # (64, 6, 6)
    nn.Dropout(0.25),

    nn.Flatten(),
    nn.Linear(64 * 6 * 6, 512),      # there are 64 output maps, each of size 6x6
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

# Place the model on the GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

def evaluate_model(model, data_loader):
    """
    :param model: the model to evaluate
    :param data_loader: an iterator over the data
    :return: percentage of correct model predictions
    """
    correct, total = 0, 0

    with torch.no_grad():  # disable gradient calculation
        # Iterate over the data
        for data in data_loader:
            # Move the data tensors to the GPU
            x, y = data[0].to(device), data[1].to(device)

            # Compute the model predictions for the inputs
            output = model(x)
            _, y_pred = torch.max(output.data, 1)

            # Add the number of correct predictions
            total += y.size(0)
            correct += (y_pred == y).sum().item()
    return correct / total

# Store statistics during the training
train_history = []
val_history = []

# The training loop
for epoch in range(30):
    start_time = time.time()
    running_loss = 0.0

    # Iterate over the data, one batch at time
    for i, data in enumerate(train_loader, 0):
        # Move the data tensors to the GPU
        x_train, y_train = data[0].to(device), data[1].to(device)

        # Initialize the gradients
        optimizer.zero_grad()

        # Compute the predicted label for the input tensor
        y_pred = model(x_train)

        # Compute the loss
        loss = loss_fn(y_pred, y_train)

        # Compute the gradient of the loss with respect to the network parameters
        loss.backward()

        # Update the model's parameters using one step of gradient descent
        optimizer.step()

        # Print the average loss every 100 mini-batches
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch: {epoch + 1}, iteration: {i + 1}, loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    # Evaluate the model on the training and validation sets at the end of each epoch
    train_accuracy = evaluate_model(model, train_loader)
    train_history.append(train_accuracy)
    val_accuracy = evaluate_model(model, validation_loader)
    val_history.append(val_accuracy)
    elapsed_time = time.time() - start_time

    print('=' * 80)
    print(f'Epoch {epoch + 1} completed in {elapsed_time:.2f} sec, '
          f'accuracy: {train_accuracy:.4f}, val accuracy: {val_accuracy:.4f}')
    print('=' * 80)

print('Finished training\n')

# Plot the learning curve
plot_utils.plot_learning_curve(training_acc=train_history,
                               validation_acc=val_history,
                               file_name='pytorch_cnn_learning_curve')

# Evaluate the model on the test set
test_accuracy = evaluate_model(model, test_loader)
print(f'Test accuracy: {test_accuracy:.4f}')






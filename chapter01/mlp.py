"""Implementing an MLP (multi-layer perceptron) on CIFAR-10 using TensorFlow 2.0 and Keras
    Author: Roi Yehoshua
    Date: June 2020
"""
import numpy as np
from tensorflow import keras
import tensorflow.keras.layers as layers

import plot_utils

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('x_train data type:', x_train.dtype)

# Print the number of samples in each category
unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Plot the first 50 images from the training set
plot_utils.plot_images(x_train[:50], class_names, y_train[:50], file_name='cifar-10')

# Scale the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the network as an MLP with two hidden layers
model = keras.models.Sequential()
model.add(layers.Flatten(input_shape=[32, 32, 3]))
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# The same as:
# model = keras.models.Sequential([
#     layers.Flatten(input_shape=[32, 32, 3]),
#     layers.Dense(200, activation='relu'),
#     layers.Dense(100, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])

model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=30, validation_split=0.1)

# Plot the learning curve
plot_utils.plot_learning_curve(training_acc=history.history['accuracy'],
                               validation_acc=history.history['val_accuracy'],
                               file_name='mlp_learning_curve')

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test)
print(f'Test accuracy: {np.round(results[1], 4)}')

# Using the model to make predictions
y_prob = model.predict(x_test[:5])
print(y_prob.round(3))

y_pred = model.predict_classes(x_test[:5])
print(y_pred)

plot_utils.plot_predictions(x_test[:5], class_names, y_test[:5],
                            y_pred, file_name='mlp_predictions')












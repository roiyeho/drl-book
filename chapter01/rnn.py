import numpy as np
from tensorflow import keras
import tensorflow.keras.layers as layers

import plot_utils

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Scale the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshaping the input tensors from 3D to 2D tensors
x_train = x_train.reshape(-1, 32, 32 * 3)
x_test = x_test.reshape(-1, 32, 32 * 3)

# Build the network
model = keras.models.Sequential([
    layers.GRU(200, input_shape=[32, 32 * 3]),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

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
                               file_name='gru_learning_curve')

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test)
print(f'Test accuracy: {np.round(results[1], 4)}')

# Using the model to make predictions
y_pred = model.predict_classes(x_test[:5])
plot_utils.plot_predictions(x_test[:5], class_names, y_test[:5],
                            y_pred, file_name='gru_predictions')
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

# Setting random seed
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load the Cifar10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Scale the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the network
model = keras.models.Sequential([
    layers.Conv2D(32, 3, padding='same', input_shape=[32, 32, 3], activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
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
pd.DataFrame(history.history).plot(figsize=(8, 5), fontsize=18)
plt.grid(True)
plt.xlabel('Epoch', fontsize=20)
plt.legend(fontsize=18)
plt.title('CNN for CIFAR-10 Classification', fontsize=20)
plt.show()

# Evaluate the model on the test set
results = model.evaluate(x_test, y_test)
results = np.round(results, 4)
print(f'Test loss: {results[0]}, test accuracy: {results[1]}')

# Using the model to make predictions
x_new = x_test[:5]
y_prob = model.predict(x_new)
print(y_prob.round(3))

y_pred = model.predict_classes(x_new)
print(y_pred)
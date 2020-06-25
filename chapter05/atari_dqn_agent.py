from dqn_agent import DQNAgent

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers

class AtariDQNAgent(DQNAgent):
    def __init__(self,
                 n_actions,
                 #learning_rate=0.00001,
                 learning_rate=0.00005,
                 gamma=0.99,
                 batch_size=32,
                 replay_buffer_size=1000000,
                 replay_start_size=50000):
        super().__init__(n_actions, learning_rate, gamma, batch_size,
                         replay_buffer_size, replay_start_size)
        self.lives = -1

    def build_model(self):
        model = keras.models.Sequential([
            layers.InputLayer(input_shape=(84, 84, 4)),
            layers.Lambda(lambda obs: tf.cast(obs, np.float32) / 255),
            layers.Conv2D(32,  kernel_size=8, strides=4, activation='relu'),
            layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'),
            layers.Conv2D(64, kernel_size=3, strides=1, activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(self.n_actions)
        ])
        return model

    def remember(self, state, action, reward, next_state, done, info):
        if self.lives == -1:
            self.lives = info['ale.lives']

        # If a life was lost treat this transition as a terminating transition
        if info['ale.lives'] < self.lives:
            done = True
            self.lives = info['ale.lives']
        self.replay_buffer.append((state, action, reward, next_state, done))

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import os

class DQNAgent:
    def __init__(self, env, config):
        """
        :param env: the gym environment where the agent will run
        :param config: the set of hyperparameters
        """
        self.env = env
        self.n_actions = env.action_space.n
        self.config = config

        # Build the Q-network
        self.q_network = self.build_network()

        # Create the target network as a copy of the Q-network
        self.target_network = keras.models.clone_model(self.q_network)

        self.optimizer = keras.optimizers.Adam(self.config.alpha)

    def build_network(self):
        c = self.config.cnn

        model = keras.Sequential()
        model.add(layers.InputLayer(input_shape=self.env.observation_space.shape))

        # Scale the input values to be between 0 and 1
        max_input = np.max(self.env.observation_space.high)
        model.add(layers.Lambda(lambda obs: tf.cast(obs, np.float32) / max_input))

        # Create the conv layers
        for filters, kernel_size, strides in zip(c.filters, c.kernel_size, c.strides):
            layer = layers.Conv2D(filters, kernel_size, strides, activation='relu')
            model.add(layer)

        model.add(layers.Flatten())
        model.add(layers.Dense(c.fc_size, activation='relu'))
        model.add(layers.Dense(self.n_actions))

        return model

    def select_action(self, observation, epsilon):
        """
            An epsilon-greedy action selection
            :param observation: the current observations
            :param epsilon: the exploration rate
            :return: an action
        """
        if np.random.rand() <= epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = self.q_network.predict(np.expand_dims(observation, axis=0))[0]
            return np.argmax(q_values)

    def train(self, observations, actions, rewards, next_observations, dones):
        # Compute the target Q values for the sampled transitions
        next_q_values = self.target_network.predict(next_observations)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.config.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            # Forward pass: compute the Q-values for the observations in the batch
            all_q_values = self.q_network(observations)

            # Mask out the Q-values for the non-chosen actions
            mask = tf.one_hot(actions, self.n_actions)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1)

            # Compute the loss between the targets and the Q-values
            loss_fn = keras.losses.Huber()
            loss = loss_fn(target_q_values, q_values)

        # Perform a gradient descent step to minimize the loss with respect
        # to the model's trainable variables
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self, folder, agent_id):
        """Save the network params to a file"""
        agent_file = os.path.join(folder, f'agent{agent_id}_network.h5')
        keras.models.save_model(self.q_network, agent_file)

    def load_model(self, folder, agent_id):
        """Load the network params from a file"""
        agent_file = os.path.join(folder, f'agent{agent_id}_network.h5')
        self.q_network = keras.models.load_model(agent_file)
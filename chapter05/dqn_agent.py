import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import os
from collections import deque
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,
                 n_actions,
                 learning_rate=0.001,
                 gamma=0.9,
                 #gamma=0.95,
                 batch_size=64,
                 replay_buffer_size=200000,
                 replay_start_size=1000
                 ):
        """
        :param n_actions: the number of possible actions
        :param learning_rate: the learning rate for the optimizer
        :param gamma: discount factor
        :param batch_size: size of a minibatch
        :param replay_buffer_size: the size of the replay memory
        :param replay_start_size: the initial size of the replay memory before learning starts
        :param target_update_interval: number of steps between consecutive updates of
                                       the target network
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size

        # Create the replay buffer
        #self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)
        self.replay_start_size = replay_start_size

        # Build the Q-network
        self.q_network = self.build_model()

        # Create the target network as a copy of the Q-network
        self.target_network = keras.models.clone_model(self.q_network)

        # Create the optimizer
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

        self.training_step = 0

    def build_model(self):
        model = keras.models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.n_actions)
        ])
        return model

    def select_action(self, state, epsilon):
        """
        An epsilon-greedy action selection
        :param state: the current state of the environment
        :param epsilon: the exploration rate
        :return: an action
        """
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_actions)
        else:
            q_values = self.q_network.predict(np.expand_dims(state, axis=0))[0]
            return np.argmax(q_values)

    def remember(self, state, action, reward, next_state, done, info):
        """Store a new transition in the replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def sample_transitions(self):
        #indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        #mini_batch = [self.replay_buffer[index] for index in indices]
        mini_batch = self.replay_buffer.sample(self.batch_size)

        states, actions, rewards, next_states, dones = [
            np.array([transition[field_index] for transition in mini_batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def train(self):
        """Perform a single training step on the network"""

        # Check that we have enough transitions in the replay buffer
        if len(self.replay_buffer) < max(self.batch_size, self.replay_start_size):
            return

        # Sample transitions from the replay buffer
        states, actions, rewards, next_states, dones = self.sample_transitions()

        # Compute the target Q values for the sampled transitions
        next_q_values = self.target_network.predict(next_states)
        max_next_q_values = np.max(next_q_values, axis=1)
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        with tf.GradientTape() as tape:
            # Forward pass: compute the Q-values for the states in the batch
            all_q_values = self.q_network(states)

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

    def save_model(self, folder, env_id):
        """Save the network params to a file"""
        agent_file = os.path.join(folder, f'{env_id}.h5')
        keras.models.save_model(self.q_network, agent_file)

    def load_model(self, folder, env_id):
        """Load the network params from a file"""
        agent_file = os.path.join(folder, f'{env_id}.h5')
        self.q_network = keras.models.load_model(agent_file)
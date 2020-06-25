import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import os

class PGAgent:
    def __init__(self,
                 n_actions,
                 learning_rate=0.001,
                 gamma=0.95
                 ):
        """
        :param n_actions: the number of possible actions
        :param learning_rate: the learning rate for the optimizer
        :param gamma: discount factor
        """
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # Build the policy network
        self.policy_network = self.build_model()

        # Create the optimizer
        self.optimizer = keras.optimizers.Adam(self.learning_rate)

    def build_model(self):
        model = keras.models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.n_actions, activation='softmax')
        ])
        return model

    def select_action(self, state):
        """
        Using the output of actor network, pick an action stochastically
        :param state: the current state of the environment
        :return: an action
        """
        state = np.expand_dims(state, axis=0)
        probs = self.policy_network.predict(state).flatten()
        action = np.random.choice(self.n_actions, p=probs)
        return action

    def train(self, states, actions, rewards):
        """
        Perform a gradient ascent step using the episode's trajectory
        :param states: set of states encountered during the episode
        :param actions: set of actions performed during the episode
        :param rewards: set of rewards obtained during the episode
        """
        # Convert the input lists into np arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        # Compute the returns
        returns = self.compute_returns(rewards)

        with tf.GradientTape() as tape:
            # Compute the action probabilities for the states in the batch
            all_probs = self.policy_network(states)

            # Mask out the probabilities of the non-chosen actions
            mask = tf.one_hot(actions, self.n_actions)
            probs = tf.reduce_sum(all_probs * mask, axis=1)

            # Define the loss as the negative of the policy gradient
            loss = -tf.reduce_mean(returns * tf.math.log(probs))

        # Perform a gradient ascent step
        gradients = tape.gradient(loss, self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    def compute_returns(self, rewards):
        """
        Compute the episode returns as discounted sums of future rewards
        :param rewards: the set of rewards obtained during the episode
        :return:
        """
        returns = np.zeros_like(rewards)
        running_sum = 0

        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.gamma * running_sum
            returns[t] = running_sum

        # Normalize the returns
        returns = (returns - np.mean(returns)) / np.std(returns)

        return returns

    def save_model(self, folder, env_id):
        """Save the network params to a file"""
        agent_file = os.path.join(folder, f'{env_id}.h5')
        keras.models.save_model(self.policy_network, agent_file)

    def load_model(self, folder, env_id):
        """Load the network params from a file"""
        agent_file = os.path.join(folder, f'{env_id}.h5')
        self.policy_network = keras.models.load_model(agent_file)
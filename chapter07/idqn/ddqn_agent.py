import numpy as np
import tensorflow as tf
from tensorflow import keras
from idqn.dqn_agent import DQNAgent

class DoubleDQNAgent(DQNAgent):
    def __init__(self, env, config):
        """
        :param env: the gym environment where the agent will run
        :param config: a set of hyperparameters
        """
        super().__init__(env, config)

    def train(self, observations, actions, rewards, next_observations, dones):
        # Use the online network to select the best actions for the next observations
        next_q_values = self.q_network.predict(next_observations)
        best_next_actions = np.argmax(next_q_values, axis=1)

        # Use the target network to estimate the Q-values of these best actions
        next_best_q_values = self.target_network.predict(next_observations)
        next_best_q_values = next_best_q_values[np.arange(len(next_best_q_values)), best_next_actions]
        target_q_values = rewards + (1 - dones) * self.config.gamma * next_best_q_values

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
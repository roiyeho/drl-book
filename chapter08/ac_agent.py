import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
import os

class ACAgent:
    def __init__(self,
                 n_actions,
                 actor_learning_rate=0.0001,
                 critic_learning_rate=0.0001,
                 gamma=0.95,
                 beta=0.01,
                 ):
        """
        :param n_actions: the number of possible actions
        :param actor_learning_rate: the learning rate for the actor
        :param critic_learning_rate: the learning rate for the critic
        :param gamma: discount factor
        :param beta: the weight of the entropy loss
        """
        self.n_actions = n_actions
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.beta = beta

        # Build the actor and critic networks
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = keras.optimizers.Adam(self.actor_learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(self.critic_learning_rate)

    def build_actor(self):
        model = keras.models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(self.n_actions, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = keras.models.Sequential([
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def select_action(self, state):
        """
        Using the output of policy network, pick an action stochastically
        :param state: the current state of the environment
        :return: an action
        """
        state = np.expand_dims(state, axis=0)
        probs = self.actor.predict(state).flatten()
        action = np.random.choice(self.n_actions, p=probs)
        return action

    def train(self, state, action, reward, next_state, done):
        """
        Train the actor and the critic networks using the current transition
        """
        state = np.expand_dims(state, axis=0)
        next_state = np.expand_dims(next_state, axis=0)

        value = self.critic(state)[0]
        next_value = self.critic(next_state)[0] * (1 - done)

        self.train_actor(state, action, reward, value, next_value)
        self.train_critic(state, reward, next_value)

    def train_actor(self, state, action, reward, value, next_value):
        td_error = reward + self.gamma * next_value - value

        with tf.GradientTape() as tape:
            # Compute the probability of the chosen action
            all_probs = self.actor(state)

            selected_action_prob = all_probs[0][action]

            # Define the loss as the negative of the policy gradient
            policy_loss = -td_error * tf.math.log(selected_action_prob)
            entropy = -tf.reduce_sum(all_probs * tf.math.log(all_probs))

            # We want to reduce the policy loss and increase the entropy
            actor_loss = policy_loss - self.beta * entropy

        # Perform a gradient ascent step
        gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))

    def train_critic(self, state, reward, next_value):
        target_value = reward + self.gamma * next_value

        with tf.GradientTape() as tape:
            # Compute the probability of the chosen action
            predicted_value = self.critic(state)[0]

            loss_fn = keras.losses.MeanSquaredError()
            critic_loss = loss_fn(target_value, predicted_value)

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

    def save_model(self, folder, env_id):
        actor_file = os.path.join(folder, f'{env_id}-actor.h5')
        keras.models.save_model(self.actor, actor_file)
        critic_file = os.path.join(folder, f'{env_id}-critic.h5')
        keras.models.save_model(self.critic, critic_file)

    def load_model(self, folder, env_id):
        actor_file = os.path.join(folder, f'{env_id}-actor.h5')
        self.actor = keras.models.load_model(actor_file)
        critic_file = os.path.join(folder, f'{env_id}-critic.h5')
        self.critic = keras.models.load_model(critic_file)

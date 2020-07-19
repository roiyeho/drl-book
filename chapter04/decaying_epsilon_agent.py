# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from bandit_agent import BanditAgent

class DecayingEpsilonGreedyAgent(BanditAgent):
    def __init__(self, bandit, initial_epsilon, final_epsilon, epsilon_decay):
        """
        :param bandit: an instance of a bandit problem
        :param epsilon: the exploration rate
        """
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay
        super().__init__(bandit)

    def reset(self):
        super().reset()
        self.epsilon = self.initial_epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # Exploration: select a random arm
            action = np.random.randint(self.bandit.n_arms)
        else:
            # Exploitation: choose the currently best arm
            action = np.argmax(self.Q)
        self.update_exploration_rate()
        return action

    def update_exploration_rate(self):
        """Decay epsilon after each action selection"""
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay
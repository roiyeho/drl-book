# Author: Roi Yehoshua
# Date: July 2020
import numpy as np

class BanditAgent:
    def __init__(self, bandit):
        """
        :param bandit: an instance of a bandit problem
        """
        self.bandit = bandit
        self.N = np.zeros(bandit.n_arms)  # Counts the number of times an action was chosen
        self.Q = np.zeros(bandit.n_arms)  # Estimated values of the actions

    def select_action(self):
        """Override this method to define the action selection rule"""
        raise NotImplementedError

    def update_q(self, action, reward):
        """Update the Q-value of the chosen action"""
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])




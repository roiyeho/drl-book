# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from abc import ABC, abstractmethod

class BanditAgent(ABC):
    def __init__(self, bandit):
        """
        :param bandit: an instance of MultiArmedBandit
        """
        self.bandit = bandit
        self.reset()

    def reset(self):
        """Initialize the data structures"""
        self.N = np.zeros(self.bandit.n_arms)  # The number of times each action was chosen
        self.Q = np.zeros(self.bandit.n_arms)  # Estimated values of the actions

    @abstractmethod
    def select_action(self):
        """Override this method to define the action selection rule"""
        pass

    def update_q(self, action, reward):
        """Update the estimated Q-value of the chosen action
        :param action: index of the chosen arm
        :param reward: reward received
        """
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])




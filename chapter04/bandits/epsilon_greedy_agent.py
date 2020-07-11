# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from bandits.bandit_agent import BanditAgent

class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, bandit, epsilon):
        """
        :param bandit: an instance of a bandit problem
        :param epsilon: the exploration rate
        """
        super().__init__(bandit)
        self.epsilon = epsilon

    def select_action(self):
        if np.random.rand() < self.epsilon:
            # Select a random arm
            return np.random.randint(self.bandit.n_arms)
        else:
            # Exploit the currently best arm
            return np.argmax(self.Q)
# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from bandits.bandit_agent import BanditAgent

class UCBAgent(BanditAgent):
    def __init__(self, bandit, c=1):
        """
        :param bandit: an instance of a bandit problem
        :param c: degree of exploration
        """
        super().__init__(bandit)
        self.c = c
        self.t = 0

    def select_action(self):
        self.t += 1
        action = np.argmax(self.Q + self.c *
                           np.sqrt(np.log(self.t) / (self.N + 1)))
        return action
# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from bandit_agent import BanditAgent

class UCBAgent(BanditAgent):
    def __init__(self, bandit, c=1):
        """
        :param bandit: an instance of a bandit problem
        :param c: degree of exploration
        """
        self.c = c
        super().__init__(bandit)

    def reset(self):
        super().reset()
        self.step = 0

    def select_action(self):
        self.step += 1
        action = np.argmax(self.Q + self.c *
                           np.sqrt(np.log(self.step) / (self.N + 1)))
        return action
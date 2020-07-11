# Author: Roi Yehoshua
# Date: July 2020
import numpy as np

class MultiArmedBandit():
    def __init__(self, n_arms, prob=None):
        """
        :param n_arms: the number of arms
        :param prob: the success probability of each arm
        """
        self.n_arms = n_arms

        if prob is None:
            # If probabilities are not specified, then just pick up
            # random numbers between 0 and 1
            self.prob = np.random.random(n_arms)
        else:
            self.prob = prob

    def get_reward(self, action):
        """Return a reward based on the arm the user has chosen
        :param action: index of the chosen arm
        :return: a reward of 1 for success and 0 for failure
        """
        num = np.random.random()
        reward = 1 if (num < self.prob[action]) else 0
        return reward
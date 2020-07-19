# Author: Roi Yehoshua
# Date: July 2020
import numpy as np

class MultiArmedBandit():
    """A class that represents a multi-armed bandit problem, where each arm has a fixed
    probability of providing a reward of 1, and otherwise a reward of 0."""
    def __init__(self, n_arms, probs=None):
        """
        :param n_arms: the number of arms
        :param probs: probability of getting a reward from each arm
        """
        self.n_arms = n_arms

        if probs is None:
            # If probabilities are not specified, then pick up random numbers in [0,1]
            self.probs = np.random.random(n_arms)
        else:
            self.probs = probs

    def get_reward(self, action):
        """Return a reward based on the arm the user has chosen
        :param action: index of the chosen arm
        :return: a reward of 1 for success and 0 for failure
        """
        num = np.random.random()
        reward = 1 if num < self.probs[action] else 0
        return reward
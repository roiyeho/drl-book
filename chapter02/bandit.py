import numpy as np

class MultiArmedBandit():
    def __init__(self, k, prob=None):
        """k - number of machines
           prob - success probabilities for each machine"""
        self.k = k
        if prob is None:
            self.prob = np.random.random(k)
        else:
            self.prob = prob

    def get_reward(self, action):
        """action - the machine the player has chosen to play
           Returns reward 1 for success, 0 for failure"""
        num = np.random.random()  # [0.0, 1.0)
        reward = 1 if (num < self.prob[action]) else 0
        return reward
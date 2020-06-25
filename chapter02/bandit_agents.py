import numpy as np

class BanditAgent:
    def __init__(self, bandit):
        """
        :param bandit: an instance of a bandit problem
        """
        self.bandit = bandit
        self.N = np.zeros(bandit.k) # Number of times action was chosen
        self.Q = np.zeros(bandit.k) # Estimated values

    def get_action(self):
        """Override this method to define the action selection rule"""
        raise NotImplementedError

    def update_Q(self, action, reward):
        """Update the Q-value of the chosen action"""
        self.N[action] += 1
        self.Q[action] += (1 / self.N[action]) * (reward - self.Q[action])

class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, bandit, epsilon):
        """
        :param bandit: an instance of a bandit problem
        :param epsilon: the probability of exploring at each time step
        """
        super().__init__(bandit)
        self.epsilon = epsilon

    def get_action(self):
        if np.random.rand() < self.epsilon:
            # Explore a random machine
            action = np.random.randint(self.bandit.k)
        else:
            # Exploit the currently best action (break ties randomly)
            max_Q = self.Q.max()
            greedy_actions = np.nonzero(np.isclose(self.Q, max_Q))[0]
            action = np.random.choice(greedy_actions)
        return action

class UCBAgent(BanditAgent):
    def __init__(self, bandit, c=1):
        """
        :param bandit: an instance of a bandit problem
        :param c: degree of exploration
        """
        super().__init__(bandit)
        self.c = c
        self.t = 0

    def get_action(self):
        self.t += 1
        action = np.argmax(self.Q + self.c *
                           np.sqrt(np.log(self.t) / (self.N + 1)))
        return action
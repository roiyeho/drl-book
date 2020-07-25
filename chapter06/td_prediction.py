# Author: Roi Yehoshua
# Date: July 2020
from collections import defaultdict

class TDPrediction():
    """Temporal difference prediction for estimating the state value function"""
    def __init__(self, env, policy, gamma, alpha, n_episodes, max_episode_len=None):
        """
        :param env: an instance of gym environment
        :param policy: an object that implements a get_action() method
        :param gamma: the discount factor
        :param alpha: learning rate
        :param n_episodes: number of episodes to sample
        :param max_episode_len: maximum number of steps per episode
        """
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len

        self.V = defaultdict(lambda: 0)   # the value function

    def estimate_value(self):
        """Estimate the state value function of the policy
        :return: the value function
        """
        for episode in range(self.n_episodes):
            done = False
            step = 0

            state = self.env.reset()
            while not done:
                action = self.policy.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.update_v(state, reward, next_state)
                state = next_state

                step += 1
                if self.max_episode_len and step > self.max_episode_len:
                    break

            # Print out which episode we're on
            if (episode + 1) % 1000 == 0:
                print(f'\rEpisode {episode + 1}/{self.n_episodes}', end='')
        return self.V

    def update_v(self, state, reward, next_state):
        """Update the V table using the given the current transition
        :param
        """
        self.V[state] += self.alpha * (reward + self.gamma * self.V[next_state] - self.V[state])

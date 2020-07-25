# Author: Roi Yehoshua
# Date: July 2020
from collections import defaultdict

class MCPrediction():
    """Monte Carlo prediction for estimating the state value function"""
    def __init__(self, env, policy, gamma, n_episodes, max_episode_len=None):
        """
        :param env: an instance of gym environment
        :param policy: an object that implements a get_action() method
        :param gamma: the discount factor
        :param n_episodes: number of episodes to sample
        :param max_episode_len: maximum number of steps per episode
        """
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len

        self.N = defaultdict(lambda: 0)  # state visitations count
        self.returns = defaultdict(lambda: 0)  # sum of returns
        self.V = defaultdict(lambda: 0)   # the value function

    def estimate_value(self):
        """Estimate the state value function of the policy
        :return: the value function
        """
        for episode in range(self.n_episodes):
            transitions = self.run_episode()
            self.update_v(transitions)

            # Print out which episode we're on
            if (episode + 1) % 1000 == 0:
                print(f'\rEpisode {episode + 1}/{self.n_episodes}', end='')
        return self.V

    def run_episode(self):
        """Run a single episode on the environment using the given policy
        :return: a list of (state, reward) pairs
        """
        transitions = []
        done = False
        step = 0

        state = self.env.reset()
        while not done:
            action = self.policy.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            transitions.append((state, reward))
            state = next_state

            step += 1
            if self.max_episode_len and step > self.max_episode_len:
                break
        return transitions

    def update_v(self, transitions):
        """Update the V table using the given transitions
        :param transitions: list of (state, reward) pairs
        """
        G = 0  # the return (sum of discounted rewards)

        # Compute the returns backwards from the last time step to the first
        for state, reward in reversed(transitions):
            G = self.gamma * G + reward
            self.N[state] += 1
            self.returns[state] += G
            self.V[state] = self.returns[state] / self.N[state]
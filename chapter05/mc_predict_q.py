# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from collections import defaultdict

class MCPredictQ():
    """Monte-Carlo prediction for estimating the action value function"""
    def __init__(self, env, policy, gamma, n_episodes, max_episode_len):
        """
        :param env: an instance of gym environment
        :param policy: a dictionary that maps states to actions
        :param gamma: the discount factor
        :param n_episodes: number of episodes to use for evaluation
        :param max_episode_len: maximum number of steps per episode
        """
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len

        n_actions = env.action_space.n
        self.N = defaultdict(lambda: np.zeros(n_actions))  # state-action visitations count
        self.returns = defaultdict(lambda: np.zeros(n_actions))  # sum of returns
        self.Q = defaultdict(lambda: np.zeros(n_actions))  # the value function

    def predict(self, n_episodes):
        """Estimate the action value function of the policy
        :return: the value function
        """
        for episode in range(self.n_episodes):
            transitions = self.run_episode()
            self.update_q(transitions)

            if (episode + 1) % 1000 == 0:
                print(f'\rEpisode {episode + 1}/{self.n_episodes}', end='')
        return self.Q

    def run_episode(self):
        """Run a single episode on the environment using the given policy
        :return: a list of (state, action, reward) tuples
        """
        transitions = []
        state = self.env.reset()

        for step in range(self.max_episode_len):
            action = self.policy[state]
            next_state, reward, done, _ = self.env.step(action)
            transitions.append((state, action, reward))
            if done:
                break
            state = next_state
        return transitions

    def update_q(self, transitions):
        """Update the Q table using the given transitions
        :param transitions: list of (state, reward) pairs
        """
        G = 0  # the return

        # Compute the returns backwards from the last time step to the first
        for state, action, reward in reversed(transitions):
            G = self.gamma * G + reward
            self.N[state][action] += 1
            self.returns[state][action] += G
            self.Q[state][action] = self.returns[state][action] / self.N[state][action]








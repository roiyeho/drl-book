# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
from collections import defaultdict

class MCControl():
    """Monte Carlo control for finding an optimal policy"""
    def __init__(self, env, epsilon, gamma, n_episodes, max_episode_len=None):
        """
        :param env: an instance of gym environment
        :param epsilon: the exploration rate
        :param gamma: the discount factor
        :param n_episodes: number of episodes to sample
        :param max_episode_len: maximum number of steps per episode
        """
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len

        n_actions = env.action_space.n
        self.N = defaultdict(lambda: np.zeros(n_actions))  # state-action visitations count
        self.returns = defaultdict(lambda: np.zeros(n_actions))  # sum of returns
        self.Q = defaultdict(lambda: np.zeros(n_actions))  # the action-value function

    def find_best_policy(self):
        """Find the optimal action values and thereby the optimal policy
        :return: the optimal Q function
        """
        for episode in range(self.n_episodes):
            transitions = self.run_episode()
            self.update_q(transitions)

            # Print out which episode we're on
            if (episode + 1) % 1000 == 0:
                print(f'\rEpisode {episode + 1}/{self.n_episodes}', end='')
        return self.Q

    def run_episode(self):
        """Run a single episode on the environment using the current policy
        :return: a list of (state, action, reward) tuples
        """
        transitions = []
        done = False
        step = 0

        state = self.env.reset()
        while not done:
            # Sample an action from our policy
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            transitions.append((state, action, reward))
            state = next_state

            step += 1
            if self.max_episode_len and step > self.max_episode_len:
                break
        return transitions

    def get_action(self, state):
        """Use an epsilon-greedy policy to select an action
        :param state: the current state
        :return: the selected action
        """
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_q(self, transitions):
        """Update the Q table using the episode's transitions
        :param transitions: list of (state, action, reward) tuples
        """
        G = 0  # the return (sum of discounted rewards)

        # Compute the returns backwards from the last time step to the first
        for state, action, reward in reversed(transitions):
            # Update the return
            G = self.gamma * G + reward
            self.N[state][action] += 1
            self.returns[state][action] += G

            # Update the Q function, which also improves our current policy
            self.Q[state][action] = self.returns[state][action] / self.N[state][action]








import numpy as np
from collections import defaultdict

class MCControl():
    def __init__(self, env, epsilon, gamma):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.N = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns = defaultdict(lambda: np.zeros(env.action_space.n))
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def find_best_policy(self, n_episodes):
        for i in range(1, n_episodes + 1):
            if i % 1000 == 0:
                print(f'\rEpisode {i}/{n_episodes}', end='')
            episode = self.run_episode()
            self.update_Q(episode)
        return self.Q

    def run_episode(self):
        """Run a single episode on the environment using the given policy. Returns a list of all the
           states, actions, and rewards of each time step in the episode."""
        episode = []
        state = self.env.reset()

        while True:
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        return episode

    def get_action(self, state):
        """ Implement an epsilon-greedy policy based on the current Q values"""
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.env.action_space.n)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q(self, episode):
        G = 0
        returns = {}

        # Compute the returns backwards from the last time step to the first
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            # Backing up replaces (s, a) eventually, so we get first-visit returns
            returns[(s, a)] = G

        for (s, a), G in returns.items():
            self.N[s][a] += 1
            self.returns[s][a] += G
            self.Q[s][a] = self.returns[s][a] / self.N[s][a]






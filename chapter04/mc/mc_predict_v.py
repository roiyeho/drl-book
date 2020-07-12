import numpy as np
from collections import defaultdict

class MCPredictV():
    def __init__(self, env, policy, gamma=0.95, n_episodes=10000, max_episode_len=100):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.n_episodes = n_episodes
        self.max_episode_len = max_episode_len

        self.visits = defaultdict(lambda: 0)
        self.returns = defaultdict(lambda: 0)
        self.V = defaultdict(lambda: 0)

    def predict(self):
        for i in range(1, self.n_episodes + 1):
            if i % 1000 == 0:
                print(f'\rEpisode {i}/{self.n_episodes}', end='')
            transitions = self.run_episode()
            self.update_v(transitions)
        return self.V

    def run_episode(self):
        """Run a single episode on the environment using the given policy. Returns a list of all the
           states, actions, and rewards of each time step in the episode."""
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

    def update_v(self, transitions):
        G = 0  # the return
        returns = {}

        # Compute the returns backwards from the last time step to the first
        for state, action, reward in reversed(transitions):
            G = self.gamma * G + reward
            # Backing up replaces (s, a) eventually, so we get first-visit return
            returns[state] = G

        for state, G in returns.items():
            self.visits[state] += 1
            self.returns[state] += G
            self.V[state] = self.returns[state] / self.visits[state]
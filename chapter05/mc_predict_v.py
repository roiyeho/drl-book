# Author: Roi Yehoshua
# Date: July 2020
from collections import defaultdict

class MCPredictV():
    """Monte-Carlo prediction for estimating the state value function"""
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

        self.N = defaultdict(lambda: 0)  # state visitations count
        self.returns = defaultdict(lambda: 0)  # sum of returns
        self.V = defaultdict(lambda: 0)   # the value function

    def predict(self):
        """Estimate the state value function of the policy
        :return: the value function
        """
        for episode in range(self.n_episodes):
            transitions = self.run_episode()
            self.update_v(transitions)

            if (episode + 1) % 1000 == 0:
                print(f'\rEpisode {episode + 1}/{self.n_episodes}', end='')
        return self.V

    def run_episode(self):
        """Run a single episode on the environment using the given policy
        :return: a list of (state, reward) pairs
        """
        transitions = []
        state = self.env.reset()

        for step in range(self.max_episode_len):
            action = self.policy[state]
            next_state, reward, done, _ = self.env.step(action)
            transitions.append((state, reward))
            if done:
                break
            state = next_state
        return transitions

    def update_v(self, transitions):
        G = 0  # the return

        # Compute the returns backwards from the last time step to the first
        for state, reward in reversed(transitions):
            G = self.gamma * G + reward
            self.N[state] += 1
            self.returns[state] += G
            self.V[state] = self.returns[state] / self.N[state]
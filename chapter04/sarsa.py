from rl_agent import RLAgent

class SARSAAgent(RLAgent):
    def __init__(self, env, alpha=0.1, gamma=0.8, epsilon=0.5):
        super().__init__(env, alpha, gamma, epsilon)

    def update_Q(self, state, action, reward, next_state):
        next_action = self.select_action(next_state)
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
                                (reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action])
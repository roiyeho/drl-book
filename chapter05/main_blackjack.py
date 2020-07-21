import gym
from mc_predict_v import MCPredictV
from plot_utils import plot_blackjack_values

class StickOn17Policy():
    """A simple blackjack policy of sticking on any sum of 17 or greater"""
    STICK, HIT = 0, 1

    def get_action(self, state):
        if state[0] >= 17:  # state[0] is the player's sum of cards
            return self.STICK
        return self.HIT

env = gym.make('Blackjack-v0')
#state = env.reset()
#print(state)

policy = StickOn17Policy()
blackjack_mc = MCPredictV(env, policy, gamma=1, n_episodes=500000, max_episode_len=10)

V = blackjack_mc.predict()
plot_blackjack_values(V)








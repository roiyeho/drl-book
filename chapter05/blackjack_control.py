import gym
from mc_control import MCControl
import plot_utils

env = gym.make('Blackjack-v0')
blackjack_control = MCControl(env, epsilon=0.1, gamma=1, n_episodes=500000)
Q = blackjack_control.find_best_policy()
plot_utils.plot_blackjack_policy(Q, filename='blackjack_control_policy')
import gym
from mc_control import MCControl
from plot_utils import plot_policy

def blackjack_control():
    env = gym.make('Blackjack-v0')

    blackjack_control = MCControl(env, epsilon=0.1, gamma=1)
    Q = blackjack_control.find_best_policy(n_episodes=100000)

    plot_policy(Q, filename='MC_Control.png')

blackjack_control()





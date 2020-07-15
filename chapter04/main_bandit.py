# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
import matplotlib.pyplot as plt

from bandits.mab import MultiArmedBandit
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
from bandits.decaying_epsilon_agent import DecayingEpsilonGreedyAgent
from bandits.ucb_agent import UCBAgent
import plot_utils

def run_bandit(bandit, agent, n_steps):
    """Play a multi-armed bandit game with the given agent
    :param bandit: an instance of the bandit problem
    :param agent: an agent trying to solve the problem
    :param n_steps: number of time steps until the game finishes
    :return: lists of the agent's chosen actions and the observed rewards
    """
    rewards, actions = [], []

    for _ in range(n_steps):
        # Select an arm and observe the reward
        action = agent.select_action()
        reward = bandit.get_reward(action)

        # Update the action value estimates
        agent.update_q(action, reward)

        # Store the reward and the selected action
        rewards.append(reward)
        actions.append(action)
    return rewards, actions

def run_experiment(bandit, agent, n_games=1000, n_steps=1000):
    """Run a series of bandit games and average the results
    :param bandit: an instance of the bandit problem
    :param agent: an agent trying to solve the problem
    :param n_games: number of games to run
    :param n_steps: number of time steps in each game
    :return: average rewards and average count of the actions
    """
    # Store the rewards and actions of all the games
    reward_history = np.zeros(n_steps)
    action_history = np.zeros((n_steps, bandit.n_arms))

    for i in range(n_games):
        agent.reset()
        rewards, actions = run_bandit(bandit, agent, n_steps)
        print(f'Game {i + 1}/{n_games}, average reward: {np.mean(rewards)}')

        # Add the game rewards and actions to history
        reward_history += rewards
        for k, action in enumerate(actions):
            action_history[k][action] += 1

    # Compute the average of rewards in every step across the games
    reward_history /= n_games
    return reward_history, action_history

# Create a multi-armed bandit problem
bandit_probs = [0.25, 0.30, 0.65, 0.45, 0.80, 0.40, 0.10, 0.75, 0.55, 0.50]
bandit = MultiArmedBandit(n_arms=10, probs=bandit_probs)
print('Bandit probabilities:', bandit_probs)

# print('Running an epsilon-greedy agent')
# agent = EpsilonGreedyAgent(bandit, epsilon=0.1)
# rewards, actions = run_experiment(bandit, agent)
#
# plot_utils.plot_rewards(rewards, file_name='epsilon_greedy_rewards')
# plot_utils.plot_actions(actions, n_arms=10, n_games=1000,
#                         file_name='epsilon_greedy_actions')

print('Running an epsilon decay agent')
agent = DecayingEpsilonGreedyAgent(bandit, initial_epsilon=1, final_epsilon=0, epsilon_decay=0.99)
rewards, actions = run_experiment(bandit, agent, n_steps=1000)

plot_utils.plot_rewards(rewards, file_name='decaying_epsilon_rewards')
plot_utils.plot_actions(actions, n_arms=10, n_games=1000,
                        file_name='decaying_epsilon_actions')

for i in range(len(rewards)):
    print(i, rewards[i])

for i in range(len(actions)):
    print(i, actions[i])


# c = 0.5
# print(f'Running multi-armed bandit with k = {k} and c = {c}')
# agent = UCBAgent(bandit, c)
#
# # Run the experiments
# # Initialize the agent in the beginning of each experiment
#         #         agent = EpsilonGreedyAgent(bandit, epsilon)
#         #         # agent = UCBAgent(bandit, c)

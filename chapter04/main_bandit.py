# Author: Roi Yehoshua
# Date: July 2020
import numpy as np
import matplotlib.pyplot as plt

from bandits.mab import MultiArmedBandit
from bandits.epsilon_greedy_agent import EpsilonGreedyAgent
from bandits.ucb_agent import UCBAgent
import plot_utils

def run_experiment(bandit, agent, n_steps=1000):
    """
    :param bandit: an instance of the bandit problem
    :param agent: an agent trying to solve the problem
    :param n_steps: the number of time steps until the game finishes
    :return: a list of the  chosen actions and the rewards
    """
    rewards = []
    actions = []

    for _ in range(n_steps):
        # Select an arm and observe the reward
        action = agent.select_action()
        reward = bandit.get_reward(action)

        # Update the action value estimates
        agent.update_q(action, reward)

        # Add the reward and action to the stats arrays
        actions.append(action)
        rewards.append(reward)
    return actions, rewards

# Create a multi-armed bandit problem
bandit_probs = [0.25, 0.30, 0.65, 0.45, 0.80, 0.40, 0.10, 0.75, 0.55, 0.50]
n_arms = len(bandit_probs)
bandit = MultiArmedBandit(n_arms, bandit_probs)

epsilon = 0.1
print(f'Running multi-armed bandit with k = {k} and epsilon = {epsilon}')
c = 0.5
print(f'Running multi-armed bandit with k = {k} and c = {c}')

# Run the experiments
n_experiments = 100
n_steps = 2000

# Store the rewards and actions of all the experiments
reward_history = np.zeros(n_steps)
action_history = np.zeros((n_steps, n_arms))

for i in range(n_experiments):
    # Initialize the agent in the beginning of each experiment
    agent = EpsilonGreedyAgent(bandit, epsilon)
    #agent = UCBAgent(bandit, c)

    rewards, actions = run_experiment(bandit, agent, n_steps)
    print(f'Experiment {i + 1}/{n_experiments}, average reward: {np.mean(rewards)}',)

    # Add this experiment rewards to the reward history
    reward_history += rewards

    #
    for k, action in enumerate(actions):
        action_history[k][action] += 1

# Plot the rewards average
reward_history = reward_history / n_experiments
plot_utils.plot_rewards(reward_history, 'epsilon_greedy_rewards')

# Plot action history experiment-averaged
plt.figure(figsize=(12, 8))
for i in range(k):
    action_history_sum_plot = 100 * action_history_sum[:, i] / n_experiments
    plt.plot(action_history_sum_plot,
             linewidth=5,
             label=f'Arm #{i + 1}')
#plt.title(f'Action history averaged over {n_experiments} '
#          f'experiments ($\epsilon = {epsilon}$)', fontsize=20)
plt.title(f'Action history averaged over {n_experiments} '
          f'experiments (c = {c})', fontsize=20)

plt.xlabel('Time step', fontsize=18)
plt.ylabel('Action choices (%)', fontsize=18)
plt.legend(shadow=True, fontsize=13)
plt.ylim([0, 100])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

output_file = 'figures/epsilon_greedy_actions.png'
plt.savefig(output_file)


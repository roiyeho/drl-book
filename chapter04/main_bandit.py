import numpy as np
import matplotlib.pyplot as plt

from bandit import MultiArmedBandit
from bandit_agents import EpsilonGreedyAgent, UCBAgent

def run_experiment(bandit, agent, n_steps=1000):
    """
    :param bandit: an instance of the bandit problem
    :param agent: the agent trying to solve the problem
    :param n_steps: the number of time steps until the game finishes
    :return: the history of the chosen actions and the rewards
    """
    action_history = []
    reward_history = []

    for _ in range(n_steps):
        # Choose action from agent (from current Q estimate)
        action = agent.get_action()

        # Pick up reward from bandit for chosen action
        reward = bandit.get_reward(action)

        # Update Q action-value estimates
        agent.update_Q(action, reward)

        # Append to history
        action_history.append(action)
        reward_history.append(reward)
    return np.array(action_history), np.array(reward_history)

if __name__ == '__main__':
    # Bandit probabilities of success
    bandit_probs = [0.25, 0.30, 0.65, 0.45, 0.80,
                    0.40, 0.10, 0.75, 0.55, 0.50]
    k = len(bandit_probs)

    # Initialize bandit
    bandit = MultiArmedBandit(k, bandit_probs)

    #epsilon = 0.1
    #print(f'Running multi-armed bandit with k = {k} and epsilon = {epsilon}')
    c = 0.5
    print(f'Running multi-armed bandit with k = {k} and c = {c}')

    # Run the experiments
    n_experiments = 1000
    n_steps = 2000

    # Store the reward and action histories of all experiments
    reward_history_sum = np.zeros(n_steps)
    action_history_sum = np.zeros((n_steps, k))

    for i in range(n_experiments):
        # Initialize the agent in the beginning of each experiment
        #agent = EpsilonGreedyAgent(bandit, epsilon)
        agent = UCBAgent(bandit, c)

        action_history, reward_history = run_experiment(bandit, agent, n_steps)
        print(f'Experiment {i + 1}/{n_experiments}')
        print(f'Average reward = {np.mean(reward_history)}')

        # Sum up experiment reward (later to be divided to represent an average)
        reward_history_sum += reward_history

        # Sum up action history
        for j, action in enumerate(action_history):
            action_history_sum[j][action] += 1

    # Plot reward history experiment-averaged
    reward_history_avg = reward_history_sum / np.float(n_experiments)

    interval = 10
    x = range(0, n_steps, interval)
    plt.plot(x, reward_history_avg[::interval])
    plt.xlabel('Time step')
    plt.ylabel('Average reward')
    #plt.title(f'Reward history averaged over {n_experiments} '
    #          f'experiments ($\epsilon = {epsilon}$)')
    plt.title(f'Reward history averaged over {n_experiments} '
              f'experiments (c = {c})')

    output_file = 'output/BanditRewards.png'
    plt.savefig(output_file)

    # Plot action history experiment-averaged
    plt.figure(figsize=(12, 8))
    for i in range(k):
        action_history_sum_plot = 100 * action_history_sum[:, i] / n_experiments
        plt.plot(action_history_sum_plot,
                 linewidth=5,
                 label=f'Machine #{i + 1}')
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

    output_file = 'output/BanditActions.png'
    plt.savefig(output_file)


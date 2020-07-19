import matplotlib.pyplot as plt

def plot_rewards(rewards, file_name, display_interval=10):
    """Plot average reward for each time step
    :param rewards: reward received at each step
    :param file_name: the file where the figure will be saved
    """
    x = range(1, len(rewards) + 1, display_interval)
    plt.plot(x, rewards[::display_interval])
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.savefig(f'figures/{file_name}.png')
    plt.close()

def plot_actions(actions, n_arms, n_games, file_name):
    """Plot the average number of times each action was chosen"""
    for i in range(n_arms):
        # Compute the average number of times each action was chosen in each step
        action_count_avg = 100 * actions[:, i] / n_games
        plt.plot(action_count_avg, linewidth=2, label=f'Arm {i + 1}')

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('% of choosing the action', fontsize=12)
    plt.legend(shadow=True)
    plt.ylim([0, 100])
    plt.savefig(f'figures/{file_name}.png')
    plt.close()
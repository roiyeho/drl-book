import matplotlib.pyplot as plt
import os
import numpy as np

def plot_rewards(results_folder, rewards, title=''):
    x = range(0, len(rewards))
    plt.plot(x, rewards)
    plt.title(title)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total reward', fontsize=12)
    graph_file = os.path.join(results_folder, f'rewards_graph.png')
    plt.savefig(graph_file)
    plt.close()

def plot_steps(results_folder, steps, title=''):
    x = range(0, len(steps))
    plt.plot(x, steps)
    plt.title(title)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    graph_file = os.path.join(results_folder, f'steps_graph.png')
    plt.savefig(graph_file)
    plt.close()

def plot_rewards_from_file(file_path, delimiter=',', display_interval=10, title=''):
    results = np.loadtxt(file_path, skiprows=1, delimiter=delimiter)
    plt.plot(results[::display_interval, 0], results[::display_interval, 1])
    plt.title(title)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Total reward', fontsize=12)
    plt.show()

def plot_steps_from_file(file_path, delimiter=',', display_interval=10, title=''):
    results = np.loadtxt(file_path, skiprows=1, delimiter=delimiter)
    plt.plot(results[::display_interval, 0], results[::display_interval, 2])
    plt.title(title)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Steps', fontsize=12)
    plt.show()
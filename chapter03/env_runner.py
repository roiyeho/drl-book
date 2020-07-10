# Author: Roi Yehoshua
# Date: June 2020
import gym
import pickle

from grid_world import GridWorldEnv
from grid_world_config import GridWorldEnvConfig

def run_environment(env, n_episodes=10, max_episode_len=100):
    """ Run a series of episodes on a given environment using random actions
    :param env: instance of a gym environment
    :param n_episodes: number of episodes
    :param max_episode_len: maximum number of time steps in episode
    """
    for episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0

        for step in range(max_episode_len):
            env.render()

            # Choose an action (here the agent's code should be implemented)
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                env.render()
                break

        print(f'Episode {episode + 1} finished after {step + 1} steps '
              f'with total reward {total_reward:.3f}')

def run_policy(env, policy, n_episodes=10, max_episode_len=100):
    """Run a given policy on a gym environment
    :param env: instance of a gym environment
    :param policy: a dictionary that maps states to actions
    :param n_episodes: number of episodes
    :param max_episode_len: maximum number of time steps in episode
    :return:
    """
    for episode in range(n_episodes):
        observation = env.reset()
        total_reward = 0

        for step in range(max_episode_len):
            env.render()

            action = policy[observation]
            observation, reward, done, info = env.step(action)
            total_reward += reward

            if done:
                env.render()
                break

        print(f'Episode {episode + 1} finished after {step + 1} steps '
              f'with total reward {total_reward:.3f}')

#env = gym.make('Taxi-v3')
env = GridWorldEnv(config=GridWorldEnvConfig())
#run_environment(env, n_episodes=1)

with open('policy.h5', 'rb') as file:
    policy = pickle.load(file)
run_policy(env, policy, n_episodes=1)


SEED = 0

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import gym
from dqn_agent import DQNAgent
from env_runner import EnvRunner

env = gym.make('CartPole-v0')
env.seed(SEED)

agent = DQNAgent(n_actions=env.action_space.n)
runner = EnvRunner(env, agent)

#runner.run()

#runner.make_video()

SEED = 0

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import gym
from ac_agent import ACAgent
from ac_env_runner import ACEnvRunner

env = gym.make('CartPole-v0')
env.seed(SEED)

agent = ACAgent(n_actions=env.action_space.n)
runner = ACEnvRunner(env, agent)

runner.run()
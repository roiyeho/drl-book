SEED = 0

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import gym
from pg_agent import PGAgent
from env_runner import EnvRunner

env = gym.make('CartPole-v0')
env.seed(SEED)

agent = PGAgent(n_actions=env.action_space.n)
runner = EnvRunner(env, agent)

runner.run()
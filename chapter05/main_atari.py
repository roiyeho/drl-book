SEED = 0

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

import gym

from gym.wrappers import atari_preprocessing
from atari_wrappers import FireOnResetWrapper, FrameStackWrapper
from atari_dqn_agent import AtariDQNAgent
from env_runner import EnvRunner

env = gym.make('BreakoutNoFrameskip-v4')
env.seed(SEED)
env = atari_preprocessing.AtariPreprocessing(env)
#env = FireOnResetWrapper(env)
env = FrameStackWrapper(env, n_frames=4)

test_env = gym.make('BreakoutNoFrameskip-v4')
test_env.seed(SEED)
test_env = atari_preprocessing.AtariPreprocessing(test_env, noop_max=0)
#test_env = FireOnResetWrapper(test_env)
test_env = FrameStackWrapper(test_env, n_frames=4)

agent = AtariDQNAgent(n_actions=env.action_space.n)
runner = EnvRunner(env,
                   agent,
                   n_episodes=20000,
                   exploration_steps=200000,
                   train_interval=4,
                   target_update_interval=10000,
                   initial_epsilon=1,
                   final_epsilon=0.01,
                   test_env=test_env,
                   check_solved=False
                   )

runner.run()
#runner.plot_results_from_file('results/BreakoutNoFrameskip-v4-20200421-034803.txt')
#runner.make_video()
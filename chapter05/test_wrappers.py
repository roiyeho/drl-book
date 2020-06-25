import matplotlib.pyplot as plt
import gym

from gym.wrappers import atari_preprocessing
from atari_wrappers import FireOnResetWrapper, FrameStackWrapper

env = gym.make('BreakoutNoFrameskip-v4')
print('State space:', env.observation_space)
print('Action space:', env.action_space)
print(env.get_action_meanings())

#env.render()
#input()

obs = env.reset()
plt.imshow(obs)
plt.show()
plt.clf()

env = atari_preprocessing.AtariPreprocessing(env)
obs = env.reset()
plt.imshow(obs, cmap='gray')
plt.show()
plt.clf()

env = FireOnResetWrapper(env)
obs = env.reset()
plt.imshow(obs, cmap='gray')
plt.show()
plt.clf()

n_frames = 3
env = FrameStackWrapper(env, n_frames=n_frames)
env.reset()

for _ in range(n_frames):
    obs, _, _, _ = env.step(3)  # Move left

plt.imshow(obs)
plt.show()

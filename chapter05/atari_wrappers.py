import numpy as np
import gym
from collections import deque

class FireOnResetWrapper(gym.Wrapper):
    """Fire (action 1), when a life was lost or the game just started, so that
    the agent doesn't stand around doing nothing."""
    def __init__(self, env, fire_action=1):
        super().__init__(env)
        self.env = env
        self.fire_action = fire_action
        self.lives = 0

    def reset(self):
        self.env.reset()
        obs, _, _, info = self.env.step(self.fire_action)
        self.lives = info['ale.lives']
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if info['ale.lives'] < self.lives:
            obs, reward, done, info = self.env.step(self.fire_action)
            self.lives = info['ale.lives']
        return obs, reward, done, info

class FrameStackWrapper(gym.ObservationWrapper):
    """Stack previous four frames"""
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.env = env
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Add a new dimension to the observation space
        old_space = env.observation_space
        shape = old_space.shape[0:2] + (n_frames, )
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

    def reset(self):
        obs = self.env.reset()
        # Add a color channel to the observation
        obs = np.expand_dims(obs, axis=2)

        # Duplicate the first frame 4 times
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return np.concatenate(self.frames, axis=2)

    def observation(self, obs):
        # Add a color channel to the observation
        obs = np.expand_dims(obs, axis=2)
        self.frames.append(obs)
        return np.concatenate(self.frames, axis=2)

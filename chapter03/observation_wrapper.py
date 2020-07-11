import gym
import cv2
import matplotlib.pyplot as plt

class ConvertToGrayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        new_observation = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        return new_observation

# Test the observation
if __name__ == '__main__':
    env = gym.make('Breakout-v4')
    # Display an image from the original environment
    observation = env.reset()
    print('Image shape before processing:', observation.shape)
    plt.imshow(observation)
    plt.show()

    # Wrap the environment with the observation wrapper
    env = ConvertToGrayscale(env)
    observation = env.reset()
    print('Image shape after processing:', observation.shape)
    plt.imshow(observation, cmap='gray')
    plt.show()



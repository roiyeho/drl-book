# Author: Roi Yehoshua
# Date: June 2020
import gym

def make_video(env, max_episode_len=500):
    """Record a video of a random agent acting in the given environment
    :param env: an instance of a gym environment
    :param max_episode_len: maximum number of steps to run
    """
    # Wrap the environment in side a Monitor
    env = gym.wrappers.Monitor(env, f'videos/{env.spec.id}')

    # Run one episode on the wrapped environment
    env.reset()
    for step in range(max_episode_len):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
    print(f'Finished recording the policy')

if __name__ == '__main__':
    env = gym.make('Breakout-v4')
    make_video(env)
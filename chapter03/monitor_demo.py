# Author: Roi Yehoshua
# Date: June 2020
import gym

def make_video(env, max_episode_len=100):
    # Add a Monitor wrapper for recording a video
    env = gym.wrappers.Monitor(env, f'videos/{env.spec.id}')
    env.reset()

    # Run one episode on the wrapped environment
    for step in range(max_episode_len):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break

    print(f'Finished recording the policy')

env = gym.make('CartPole-v0')
make_video(env)
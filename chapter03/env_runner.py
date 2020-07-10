import gym
from grid_world import GridWorldEnv
from grid_world_config import GridWorldEnvConfig

def run_environment(env, n_episodes=10, max_episode_len=100):
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

#env = gym.make('Taxi-v3')
env = GridWorldEnv(config=GridWorldEnvConfig())
run_environment(env)

import gym

env = gym.make('CartPole-v0')
#print(gym.envs.registry.all())

observation = env.reset()
print(observation)

print(env.action_space)
random_actions = [env.action_space.sample() for _ in range(5)]
print('Actions:', random_actions)

print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)

# Run the environment for 200 time steps
# env.reset()
# for _ in range(200):
#     env.render()
#     action = env.action_space.sample()
#     env.step(action)

# A typical agent-environment loop
for i_episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        if done:
            print(f'Episode finished after {t + 1} time steps')
            break
        observation = next_observation
import gym

print(gym.envs.registry.all())

env = gym.make('CartPole-v0')

observation = env.reset()
print(observation)

print(env.step(0))

print(env.action_space)
print(env.observation_space)

print(env.observation_space.low)
print(env.observation_space.high)

random_actions = [env.action_space.sample() for _ in range(5)]
print('Actions:', random_actions)

env.render()
input()


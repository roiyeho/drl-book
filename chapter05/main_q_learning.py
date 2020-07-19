import gym
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

from q_learning import QLearningAgent
from sarsa import SARSAAgent

env = gym.make('Taxi-v3')
env.seed(42)
q_learning_agent = QLearningAgent(env)
print('Q-Learning')
q_learning_rewards = q_learning_agent.train()

#q_learning_agent.display_policy()

x = range(0, 10000, 100)
plt.plot(x, q_learning_rewards[:10000:100], label='Q-Learning')
plt.xlabel('Episode')
plt.ylabel('Total reward per episode')
plt.title('Q-learning: Reward During Training')
plt.savefig('output/taxi_q_learning.png')

# Comparing SARSA to Q-learning
sarsa_agent = SARSAAgent(env)
print('SARSA')
sarsa_rewards = sarsa_agent.train()

plt.clf()
x = range(0, 10000, 100)
plt.plot(x, q_learning_rewards[:10000:100], label='Q-Learning')
plt.plot(x, sarsa_rewards[:10000:100], label='SARSA')
plt.xlabel('Episode')
plt.ylabel('Total reward per episode')
plt.title('Q-learning vs Sarsa')
plt.legend()
plt.savefig('output/taxi_q_learning_vs_sarsa.png')




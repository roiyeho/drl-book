from grid_world.env import GridWorldEnv
from grid_world.config import GridWorldEnvConfig
from grid_world.policy import GridWorldPolicy
from mc_prediction import MCPrediction

env = GridWorldEnv(GridWorldEnvConfig())
policy = GridWorldPolicy(filename='grid_world/grid_policy.h5')
mc_predict = MCPrediction(env, policy, gamma=0.95, n_episodes=100000, max_episode_len=100)

V = mc_predict.estimate_value()
print()
env.print_values(V)
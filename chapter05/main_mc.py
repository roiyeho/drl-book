import pickle

from grid_world import GridWorldEnv
from grid_world_config import GridWorldEnvConfig
from mc_predict_v import MCPredictV

env = GridWorldEnv(GridWorldEnvConfig())
with open('policy.h5', 'rb') as file:
    policy = pickle.load(file)
mc_predict = MCPredictV(env, policy, gamma=0.95, n_episodes=100000, max_episode_len=100)

V = mc_predict.predict()
print()
env.print_values(V)
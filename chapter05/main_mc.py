import pickle

from grid_world import GridWorldEnv
from grid_world_config import GridWorldEnvConfig
from mc_predict_v import MCPredictV

class GridWorldPolicy():
    def __init__(self, filename):
        with open(filename, 'rb') as file:
            self.policy = pickle.load(file)

    def get_action(self, state):
        return self.policy[state]

env = GridWorldEnv(GridWorldEnvConfig())
policy = GridWorldPolicy(filename='policy.h5')
mc_predict = MCPredictV(env, policy, gamma=0.95, n_episodes=100000, max_episode_len=100)

V = mc_predict.predict()
print()
env.print_values(V)
import pickle

from grid_world import GridWorldEnv
from grid_world_config import GridWorldEnvConfig
from mc.mc_predict_v import MCPredictV

env = GridWorldEnv(GridWorldEnvConfig())
with open('policy.h5', 'rb') as file:
    policy = pickle.load(file)
mc_predict = MCPredictV(env, policy)

V = mc_predict.predict()
print(V)
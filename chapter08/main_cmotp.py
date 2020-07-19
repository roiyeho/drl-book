SEED = 0

import numpy as np
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)

from cmotp.cmotp import CMOTP
from idqn.team_config import TeamConfig
from idqn.agent_config import AgentConfig
from idqn.env_runner import EnvRunner

env = CMOTP()

agent_config = AgentConfig(agent_type='dqn')
team_config = TeamConfig()

runner = EnvRunner(env,
                   team_config,
                   agent_config,
                   episodes=2000)

runner.run()
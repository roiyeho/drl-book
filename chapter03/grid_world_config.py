# Author: Roi Yehoshua
# Date: June 2020
class GridWorldEnvConfig():
    """Configuration settings for the grid world environment"""
    def __init__(self):
        # Grid dimensions
        self.n_rows, self.n_columns = 5, 5

        # Object locations
        self.gold_location = (2, 4)
        self.pits = [(1, 4), (2, 2)]
        self.obstacles = [(1, 1), (3, 3)]
        self.agent_init_location = (0, 0)

        # Actions
        self.n_actions = 4
        self.directions = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        self.action_meanings = ['N', 'E', 'S', 'W']
        self.action_noise = 0.2

        # Reward definitions
        self.gold_reward = 1.0
        self.pit_reward = -1.0
        self.living_reward = -0.05

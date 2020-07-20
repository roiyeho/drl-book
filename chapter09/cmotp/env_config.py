class CMOTPConfig:
    """CMOTP environment parameters"""

    def __init__(self):
        # Grid world dimensions
        self.grid_width = 16
        self.grid_height = 16
        self.grid_dimensions = (self.grid_height, self.grid_width)

        self.mid_column = self.grid_width // 2

        # Actions
        self.n_actions = 5  # 'Up':0, 'Right':1, 'Down':2, 'Left':3, 'NOOP':4

        # A dictionary that maps actions to directions (dx, dy)
        self.directions = {
            0: (0, -1),  # North
            1: (1, 0),  # East
            2: (0, 1),  # South
            3: (-1, 0),  # West
            4: (0, 0)  # No-Op
        }
        self.noop_action = 4

        # Locations of the agents. By default they are located at the bottom right
        # and left corners.
        self.n_agents = 2
        self.agents_x = [1, self.grid_width - 1]
        self.agents_y = [self.grid_height - 1, self.grid_height - 1]

        # Goods location
        self.goods_x = self.mid_column
        self.goods_y = 11

        # Drop-zone location
        self.dropzone_x = self.mid_column
        self.dropzone_y = 0

        # Reward function
        self.delivered_reward = 1.0

        # Colors
        self.agent_colors = [250.0, 200.0]  # [Agent1, Agent2]
        self.goods_color = 150.0
        self.obstacle_color = 50.0

        # Obstacle locations are defined as tuples of (y, x)
        self.obstacles = []

        # Left column is blocked
        for i in range(self.grid_height):
            self.obstacles.append((i, 0))

        # The wall above the agents
        for i in range(0, self.grid_width):
            if i != self.mid_column:
                self.obstacles.append((self.grid_height - 2, i))

        # Platform above the goods
        for i in range(5, self.grid_width - 4):
            self.obstacles.append((self.goods_y - 1, i))

        # Touchdown area
        for i in range(0, self.mid_column - 1):
            self.obstacles.append((0, i))
        for i in range(self.mid_column + 2, self.grid_width):
            self.obstacles.append((0, i))


import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
import copy
import cv2
from cmotp.env_config import CMOTPConfig

class CMOTP(gym.Env):
    """Cooperative multi-agent object transportation problem.
        Described in Palmer et al., "Lenient Multi-Agent Deep Reinforcement Learning" (2017).
    """

    def __init__(self, config=CMOTPConfig(), max_steps=10000):
        """
        :param config: an object containing the configuration to use
        :param max_steps: maximum length of an episode
        """
        self.config = config
        self.max_steps = max_steps

        self.n_agents = self.config.n_agents

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(self.config.n_actions)
        obs_shape = self.config.grid_dimensions + (1,)
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint)

        # Define the environment id
        self.spec = EnvSpec('CMOTP-v0')

    def reset(self):
        # Initialize the state matrix
        self.state = np.zeros(self.config.grid_dimensions, dtype=np.uint)

        # Set up the obstacles, goods, and agents
        self.set_obstacles()
        self.init_goods()
        self.init_agents()

        # Initialize the step counter for the episode
        self.steps = 0

        # Return the initial observations
        return self.get_observations()

    def set_obstacles(self):
        """Set the obstacle positions within the environment"""
        for y, x in self.config.obstacles:
            self.state[y][x] = self.config.obstacle_color

    def init_goods(self):
        """Initialize the goods position and carrier ids"""

        # Set the initial x and y coordinates of the goods
        self.goods_x = self.config.goods_x
        self.goods_y = self.config.goods_y

        # Store the ID of the agents on the left and right hand side of the goods
        # once it has been picked up. Initially set to -1.
        self.goods_left = -1
        self.goods_right = -1

        # Set the delivered status of the goods to be initially false
        self.delivered = False

        # Update the goods position in the state matrix
        self.state[self.goods_y][self.goods_x] = self.config.goods_color

    def init_agents(self):
        """Initialize the agents and their positions within the grid"""

        # Set the initial agent x and y positions
        self.agents_x = copy.deepcopy(self.config.agents_x)
        self.agents_y = copy.deepcopy(self.config.agents_y)

        # List used to indicate whether or not the agents are holding goods
        self.holding_goods = [False for _ in range(self.n_agents)]

        # Update the agents' positions in the state matrix
        for i in range(self.n_agents):
            self.state[self.agents_y[i]][self.agents_x[i]] = self.config.agent_colors[i]

    def get_observations(self):
        """Return an observation for each agent.
           Here the environment is fully observable, so observation=state."""
        observations = []

        for i in range(self.n_agents):
            # First copy the state so the agents don't get references to the same object
            # Then add a dummy color dimension to the observation
            obs = np.expand_dims(np.copy(self.state), axis=2)
            observations.append(obs)
        return observations

    def step(self, actions):
        """
        Change the environment state based on the agents' actions.
        :param actions: list of integers, one per agent
        """
        # Move the agents according to the selected actions
        self.move_agents(actions)

        # Check if the agents are in a position they can grasp the goods
        self.pickup_goods()

        # Check if the goods has reached a drop-zone
        self.is_goods_delivered()

        # Compute reward
        reward = self.config.delivered_reward if self.delivered else 0

        # Generate observations for each agent
        observations = self.get_observations()

        # Check task termination
        self.steps += 1
        done = self.delivered or self.steps >= self.max_steps

        return observations, reward, done, {}

    def move_agents(self, actions):
        """Move the agents according to the specified actions
        :param actions: List of integers providing an action for each agent
        """
        # Check if the two agents are holding the goods and moving in the same direction
        if self.goods_left > -1 and self.goods_right > -1 and \
                actions[self.goods_left] == actions[self.goods_right] and \
                actions[self.goods_left] != self.config.noop_action:

            dx, dy = self.config.directions[actions[self.goods_left]]

            # Compute the new positions of the agents and the goods
            target_left_x = self.agents_x[self.goods_left] + dx
            target_left_y = self.agents_y[self.goods_left] + dy
            target_right_x = self.agents_x[self.goods_right] + dx
            target_right_y = self.agents_y[self.goods_right] + dy
            target_goods_x = self.goods_x + dx
            target_goods_y = self.goods_y + dy

            # Check for any potential collisions
            collision = False
            if dx > 0 and self.is_colliding(target_right_x, target_right_y):
                collision = True
            elif dx < 0 and self.is_colliding(target_left_x, target_left_y):
                collision = True
            elif dy != 0 and (self.is_colliding(target_left_x, target_left_y) or
                              self.is_colliding(target_right_x, target_left_y) or
                              self.is_colliding(target_goods_x, target_goods_y)):
                collision = True

            # If no collisions found, move both agents and the goods in the same direction
            if not collision:
                self.move_agent(self.goods_left, target_left_x, target_left_y)
                self.move_agent(self.goods_right, target_right_x, target_right_y)
                self.move_goods(target_goods_x, target_goods_y)

        else:
            for i in range(self.n_agents):
                if not self.holding_goods[i]:
                    dx, dy = self.config.directions[actions[i]]
                    target_x = self.agents_x[i] + dx
                    target_y = self.agents_y[i] + dy
                    if not self.is_colliding(target_x, target_y):
                        self.move_agent(i, target_x, target_y)

    def move_agent(self, agent_id, target_x, target_y):
        """Move an agent to the target location
        :param agent_id: int, id of the agent
        :param target_x: int, target x coordinate
        :param target_y: int, target y coordinate
        :return:
        """
        self.state[self.agents_y[agent_id]][self.agents_x[agent_id]] -= self.config.agent_colors[agent_id]
        self.state[target_y][target_x] += self.config.agent_colors[agent_id]
        self.agents_x[agent_id] = target_x
        self.agents_y[agent_id] = target_y

    def move_goods(self, target_x, target_y):
        """Moves the goods to the given target location
        :param target_x: int, target x coordinate
        :param target_y: int, target y coordinate
        """
        self.state[self.goods_y][self.goods_x] -= self.config.goods_color
        self.state[target_y][target_x] += self.config.goods_color
        self.goods_x = target_x
        self.goods_y = target_y

    def is_colliding(self, x, y):
        """Check if cell (x,y) is inside the boundaries and is empty
        :param x: int, x coordinate
        :param y: int, y coordinate
        :return: boolean True if a collision was found
        """
        if x < 0 or x >= self.config.grid_width or y < 0 or y >= self.config.grid_height or \
                self.state[y][x] != 0:
            return True
        return False

    def pickup_goods(self):
        """Check whether the agents are in a position to pick up the goods."""

        # Check if there is an agent on the left hand side of the goods
        if self.goods_left == -1 and \
                self.state[self.goods_y][self.goods_x - 1] > 0:
            for i in range(self.n_agents):
                if self.agents_x[i] == self.goods_x - 1 \
                        and self.agents_y[i] == self.goods_y  \
                        and not self.holding_goods[i]:
                    self.goods_left = i
                    self.holding_goods[i] = True
                    break

        # Check to see if there is an agent on the right hand side of the goods
        if self.goods_right == -1 and \
                self.state[self.goods_y][self.goods_x + 1] > 0:
            for i in range(self.n_agents):
                if self.agents_x[i] == self.goods_x + 1 \
                        and self.agents_y[i] == self.goods_y \
                        and not self.holding_goods[i]:
                    self.goods_right = i
                    self.holding_goods[i] = True
                    break

    def is_goods_delivered(self):
        """Check if the goods has been placed in the drop-zone"""

        if self.goods_x == self.config.dropzone_x and self.goods_y == self.config.dropzone_y:
            self.delivered = True
            self.state[self.goods_y][self.goods_x] -= self.config.goods_color
            self.goods_y = -1

    def render(self, mode='human'):
        r = 25  # The number of times each pixel is repeated
        img = np.repeat(np.repeat(self.state, r, axis=0), r, axis=1).astype(np.uint8)

        if mode == 'human':
            cv2.imshow('CMOTP', img)
            k = cv2.waitKey(5000)
            if k == 27:  # If escape was pressed exit
                cv2.destroyAllWindows()

        return img

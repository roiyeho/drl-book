# Author: Roi Yehoshua
# Date: June 2020
import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec

class GridWorldEnv(gym.Env):
    """Implement the grid world environment described in the book"""
    def __init__(self, config):
        """Define the environment properties
        :param config (object): the environment's configuration settings
        """
        self.config = config

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(self.config.n_actions)
        self.observation_space = spaces.Discrete(self.config.n_rows * self.config.n_columns)

        # Define the environment id
        self.spec = EnvSpec('GridWorld-v0')

    def reset(self):
        # Reset the agent's location to its initial location
        self.agent_location = self.config.agent_init_location

        # Save the last chosen action for visualization purposes
        self.last_action = None

        return self.agent_location

    def step(self, action):
        """Update the environments's state based on the agent's action
        :param action: the selected action index
        :return: a tuple of (observation, reward, done, info)
        """
        self.move_agent(action)
        observation = self.agent_location
        reward = self.get_reward()
        done = self.check_termination()
        self.last_action = action

        return observation, reward, done, {}

    def move_agent(self, action):
        # Make the movement noisy
        turn_right = (action + 1) % self.config.n_actions
        turn_left = (action - 1) % self.config.n_actions
        transitions = [action, turn_right, turn_left]
        prob = self.config.action_noise
        actual_action = np.random.choice(transitions, 1,
                                         p=[1 - prob, prob / 2, prob / 2])[0]

        # Add the resulting direction to the agent's current location
        direction = self.config.directions[actual_action]
        from operator import add
        new_location = tuple(map(add, self.agent_location, direction))

        # Check for potential collisions
        if not self.is_colliding(new_location):
            self.agent_location = new_location

    def is_colliding(self, location):
        """Check if an agent's location is colliding with a wall or an obstacle
        :param location: a tuple of (row index, column index)
        :return: True if a collision was found
        """
        i, j = location
        if i < 0 or i >= self.config.n_rows or j < 0 or \
            j >= self.config.n_columns or location in self.config.obstacles:
            return True
        return False

    def get_reward(self):
        """Implement the reward function
        :return: a numeric reward
        """
        if self.agent_location == self.config.gold_location:
            return self.config.gold_reward
        elif self.agent_location in self.config.pits:
            return self.config.pit_reward
        else:
            return self.config.living_reward

    def check_termination(self):
        if self.agent_location == self.config.gold_location or \
                self.agent_location in self.config.pits:
            return True
        return False

    def render(self):
        """Print the agent's location and the last action taken"""
        if self.last_action is None:
            print(f'Location: {self.agent_location}')
        else:
            action_meaning = self.config.action_meanings[self.last_action]
            print(f'Action: {action_meaning}, location: {self.agent_location}')
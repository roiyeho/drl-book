# Author: Roi Yehoshua
# Date: June 2020

import operator
from collections import defaultdict

from mdp import MDP

class GridWorld(MDP):
    """Implement the grid world example described in the book"""
    def __init__(self, grid, initial_state, terminal_rewards, living_reward=-0.1):
        """
        :param grid: a matrix of cells, each cell can be occupied (1) or free (0)
        :param initial_state: the initial location of the agent
        :param terminal_rewards: a dictionary of (terminal state, reward)
        :param living_reward: a small negative cost for each action
        """
        self.grid = grid
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        self.terminal_rewards = terminal_rewards
        self.living_reward = living_reward

        states = self.generate_states()

        # A dictionary that maps actions to directions (dx, dy)
        self.directions = {
            0: (-1, 0),  # North
            1: (0, 1),  # East
            2: (1, 0),  # South
            3: (0, -1),  # West
        }

        super().__init__(states=states, actions=list(self.directions.keys()),
                         initial_state=initial_state, terminal_states=list(terminal_rewards.keys()))
        self.transitions = defaultdict(lambda: {})  # A cache for storing computed transitions

    def generate_states(self):
        """Every free cell in the grid is a possible state.
        :return: a list of states
        """
        states = []

        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:   # Check if this is a free (zero) cell
                    states.append((i, j))
        return states

    def T(self, state, action):
        """Implement the transition function. The agent moves to its intended direction with
        probability 0.8, and with probability 0.2 moves to one of the two perpendicular directions.
        :return: list of (probability, next_state) tuples
        """
        # Check if the transitions are already in the cache
        if state in self.transitions and action in self.transitions[state]:
            return self.transitions[state][action]

        if state not in self.terminal_states:
            turn_right = self.actions[(action + 1) % len(self.actions)]
            turn_left = self.actions[(action - 1) % len(self.actions)]

            transitions = [(0.8, self.step(state, action)),
                           (0.1, self.step(state, turn_right)),
                           (0.1, self.step(state, turn_left))]
        else:
            transitions = [(0.0, state)]

        # Store the transitions in the cache
        self.transitions[state][action] = transitions
        return transitions

    def step(self, state, action):
        """Execute an action in the given state and return the new state"""
        direction = self.directions[action]

        # Add the state and the direction tuples element-wise
        new_state = tuple(map(operator.add, state, direction))

        # Check that the agent didn't move out of boundaries
        if new_state in self.states:
            return new_state
        else:
            return state

    def R(self, state, action, next_state):
        """Implement the reward function"""
        # Check if we have reached a terminal state
        if next_state in self.terminal_states:
            return self.terminal_rewards[next_state]
        else:
            return self.living_reward

    def render(self, state=None):
        """Print the given state of the grid world
        :param state: if None, print the initial state
        """
        if state is None:
            state = self.initial_state

        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] != 0:
                    print('X', end='')
                else:
                    if state == (i, j):
                        print('A', end='')  # agent
                    elif (i, j) in self.terminal_states:
                        print('T', end='')
                    else:
                        print('.', end='')
            print()

    def print_values(self, V):
        """Print the values table"""
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:
                    print(f'{V[(i, j)]: .3f} ', end='')
                else:
                    s = '-'
                    print(f'{s:^7}', end='')
            print()

    def print_policy(self, policy):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.grid[i, j] == 0:
                    if policy[i, j] == 0:
                        print('↑', end='')
                    elif policy[i, j] == 1:
                        print('→', end='')
                    elif policy[i, j] == 2:
                        print('↓', end='')
                    elif policy[i, j] == 3:
                        print('←', end='')
                else:
                    print(' ', end='')
            print()







import operator
from mdp import MDP

DIRECTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # UP, RIGHT, DOWN, LEFT

class GridMDP(MDP):
    def __init__(self, grid, initial_state, terminals, rewards,
                 move_cost=-1, gamma=1.0):
        super().__init__(initial_state=initial_state, terminals=terminals,
                         actions=DIRECTIONS, gamma=gamma)

        self.rewards = rewards
        self.move_cost = move_cost

        # The states set consists of the free cells in the grid
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        for i in range(self.rows):
            for j in range(self.cols):
                if grid[i, j]:   # Check if this is a free cell (not zero)
                    states.add((i, j))
        self.states = states

        # Store the transition model in a dictionary of dictionaries T[s][a]
        transitions = {}
        for s in states:
            transitions[s] = {}
            for a in self.actions:
                transitions[s][a] = self.calculate_transitions(s, a)
        self.transitions = transitions

    def calculate_transitions(self, state, action):
        if state not in self.terminals:
            return [(0.8, self.move(state, action)),
                    (0.1, self.move(state, self.turn_right(action))),
                    (0.1, self.move(state, self.turn_left(action)))]
        else:
            return [(0.0, state)]

    def move(self, state, direction):
        """Return the state that results from moving in the given direction."""

        # Add the state and the direction tuples element-wise
        new_state = tuple(map(operator.add, state, direction))

        # Check that the agent didn't move out of boundaries
        if new_state in self.states:
            return new_state
        else:
            return state

    def turn_right(self, direction):
        return DIRECTIONS[(DIRECTIONS.index(direction) + 1) % len(DIRECTIONS)]

    def turn_left(self, direction):
        return DIRECTIONS[(DIRECTIONS.index(direction) - 1) % len(DIRECTIONS)]

    def T(self, state, action):
        return self.transitions[state][action]

    def R(self, state, action, next_state):
        if next_state in self.terminals:
            return self.rewards[next_state]
        else:
            return self.move_cost






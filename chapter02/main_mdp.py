# Author: Roi Yehoshua
# Date: June 2020

import numpy as np
from grid_world import GridWorld
from value_iteration import ValueIteration

# Define the grid cells
grid = np.zeros((5, 5))

# Set obstacles
grid[1, 1], grid[3, 3] = 1, 1

# Define the terminal rewards
terminal_rewards = {
    (2, 4): 1,   # gold
    (1, 4): -1,  # pit
    (3, 1): -1   # pit
}

# Create the grid MDP
grid_mdp = GridWorld(grid, initial_state=(0, 0), terminal_rewards=terminal_rewards)
grid_mdp.render()

# Run value iteration
vi = ValueIteration(grid_mdp)
print(f'Running value iteration')
vi.run()
print('\nFinal V table:')
grid_mdp.print_values(vi.V)

# Print the optimal policy
policy = vi.get_best_policy()
print('\nBest policy:')
grid_mdp.print_policy(policy)



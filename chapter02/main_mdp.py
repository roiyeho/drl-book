# Author: Roi Yehoshua
# Date: June 2020
import numpy as np
import pickle

from grid_world import GridWorld
from value_iteration import ValueIteration
from policy_iteration import PolicyIteration

# Define the grid world environment described in the book
grid = np.zeros((5, 5))

# Set obstacles
grid[1, 1], grid[3, 3] = 1, 1

# Define the terminal rewards
terminal_rewards = {
    (2, 4): 1,   # gold
    (1, 4): -1,  # pit
    (2, 2): -1   # pit
}

# Create the MDP
grid_mdp = GridWorld(grid, initial_state=(0, 0),
                     terminal_rewards=terminal_rewards, living_reward=-0.05)

# Sanity check
transitions = grid_mdp.T(grid_mdp.initial_state, action=1)
print(transitions)

# Run value iteration
# vi = ValueIteration(grid_mdp)
# print('Running value iteration')
# vi.run()
# vi.plot_learning_curve()
#
# print('\nFinal V table:')
# grid_mdp.print_values(vi.V)
#
# # Print the optimal policy
# policy = vi.get_best_policy()
# print('\nBest policy:')
# grid_mdp.print_policy(policy)

# Run policy iteration
pi = PolicyIteration(grid_mdp)
print('Running policy iteration')
pi.run()

# Print the optimal policy
print('\nBest policy:')
grid_mdp.print_policy(pi.policy)

# Save policy to file
with open('results/policy.h5', 'wb') as file:
    pickle.dump(pi.policy, file)



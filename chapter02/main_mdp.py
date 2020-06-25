import numpy as np
from grid_world import GridMDP
from value_iteration import value_iteration, best_policy

def print_values(V, grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j]:
                print(f'{V[(i, j)]:.3f} ', end='')
            else:
                s = '-'
                print(f'{s:^7}', end='')
        print()

def print_policy(pi, grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i, j]:
                if pi[i, j] == (0, 1):
                    print('→', end='')
                elif pi[i, j] == (0, -1):
                    print('←', end='')
                elif pi[i, j] == (-1, 0):
                    print('↑', end='')
                elif pi[i, j] == (1, 0):
                    print('↓', end='')
            else:
                print(' ', end='')
        print()

if __name__ == '__main__':
    # Define the grid world according to the example given in the text
    grid = np.ones((3, 4))
    grid[1, 1] = 0  # Obstacle
    terminals = [(0, 3), (1, 3)]
    rewards = {terminals[0]: -25, terminals[1]: +25}
    mdp = GridMDP(grid, initial_state=(0, 0), terminals=terminals,
                  rewards=rewards)

    epsilon = 0.01
    print(f'Running value iteration with eps={epsilon}')
    V = value_iteration(mdp, epsilon)
    print('\nFinal state values:')
    print_values(V, grid)

    pi = best_policy(mdp, V)
    print('Best policy:')
    print_policy(pi, grid)

# Author: Roi Yehoshua
# Date: June 2020
import matplotlib.pyplot as plt

class ValueIteration:
    """Implement the value iteration algorithm"""
    def __init__(self, mdp, gamma=0.95, epsilon=0.001):
        """
        :param mdp: an instance of the MDP class
        :param gamma: discount factor
        :param epsilon: maximum error in an iteration
        """
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self):
        """Run VI to find the optimal value function"""
        R, T = self.mdp.R, self.mdp.T  # Use shorter names

        # Initialize the V table
        self.V = {s: 0 for s in self.mdp.states}
        self.deltas = []  # Stores the delta in each iteration

        i = 0
        while True:
            V_copy = self.V.copy()
            delta = 0

            for state in self.mdp.states:
                if state not in self.mdp.terminal_states:  # The values of terminal states remain 0
                    self.V[state] = max(sum(prob * (R(state, action, next_state) + self.gamma * V_copy[next_state])
                                            for (prob, next_state) in T(state, action))
                                        for action in self.mdp.actions)
                    delta = max(delta, abs(self.V[state] - V_copy[state]))

            self.deltas.append(delta)
            i += 1
            print(f'Iteration {i}, delta: {delta:.6f}')
            if delta < self.epsilon:
                break

    def plot_learning_curve(self):
        x = range(1, len(self.deltas) + 1)
        plt.plot(x, self.deltas)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Delta', fontsize=12)
        plt.xticks(x)
        plt.savefig(f'figures/vi_learning_curve.png')
        plt.close()

    def get_best_policy(self):
        """Return the policy induced by the calculated V table"""
        policy = {}
        for state in self.mdp.states:
            if state not in self.mdp.terminal_states:
                policy[state] = max(self.mdp.actions,
                                    key=lambda action: self.expected_return(state, action))
        return policy

    def expected_return(self, state, action):
        """Compute the expected return of taking action a in state s"""
        return sum(prob * (self.mdp.R(state, action, next_state) + self.gamma * self.V[next_state])
                   for (prob, next_state) in self.mdp.T(state, action))
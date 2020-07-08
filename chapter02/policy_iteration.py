# Author: Roi Yehoshua
# Date: June 2020
import numpy as np
import matplotlib.pyplot as plt

class PolicyIteration:
    """Implement the policy iteration algorithm"""
    def __init__(self, mdp, gamma=0.95, epsilon=0.01):
        """
        :param mdp: an instance of the MDP class
        :param gamma: discount factor
        :param epsilon: maximum error in an iteration
        """
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self):
        """Run policy iteration to find the optimal policy"""

        # Initialize the policy with random actions
        self.policy = {state: np.random.randint(len(self.mdp.actions)) for state in self.mdp.states}

        # Run policy evaluation and improvement steps until no change of the policy is detected
        policy_changed = True
        i = 1
        while policy_changed:
            n_iterations = self.policy_evaluation()
            print(f'Policy evaluation {i}: {n_iterations} iterations')
            policy_changed = self.policy_improvement()
            i += 1

    def policy_evaluation(self):
        R, T = self.mdp.R, self.mdp.T  # Use shorter names

        # Initialize the V table
        self.V = {s: 0 for s in self.mdp.states}

        i = 0
        while True:
            V_copy = self.V.copy()
            delta = 0

            for state in self.mdp.states:
                if state not in self.mdp.terminal_states:
                    action = self.policy[state]
                    self.V[state] = sum(prob * (R(state, action, next_state) + self.gamma * V_copy[next_state])
                                        for (prob, next_state) in T(state, action))
                    delta = max(delta, abs(self.V[state] - V_copy[state]))

            i += 1
            if delta < self.epsilon:
                return i

    def policy_improvement(self):
        R, T = self.mdp.R, self.mdp.T  # Use shorter names
        changed = False

        for state in self.mdp.states:
            if state not in self.mdp.terminal_states:
                new_action = np.argmax([sum(prob * (R(state, action, next_state) + self.gamma * self.V[next_state])
                                            for (prob, next_state) in T(state, action))
                                        for action in self.mdp.actions])
                if new_action != self.policy[state]:
                    changed = True
                    self.policy[state] = new_action
        return changed

    def plot_learning_curve(self):
        x = range(1, len(self.deltas) + 1)
        plt.plot(x, self.deltas)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Delta', fontsize=12)
        plt.xticks(x)
        plt.savefig(f'figures/vi_learning_curve.png')
        plt.close()

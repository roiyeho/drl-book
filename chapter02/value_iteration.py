# Author: Roi Yehoshua
# Date: June 2020

class ValueIteration:
    """Implement the value iteration algorithm for solving an MDP"""
    def __init__(self, mdp, gamma=0.95, epsilon=0.001):
        """
        :param mdp: an instance of the MDP class
        :param gamma: discount factor
        :param epsilon:
        """
        self.mdp = mdp
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize the V table
        self.V = {s: 0 for s in mdp.states}

    def run(self):
        """Run VI to find the optimal value function"""
        R, T = self.mdp.R, self.mdp.T

        i = 0
        while True:
            V_copy = self.V.copy()
            delta = 0

            for state in self.mdp.states:
                if state not in self.mdp.terminal_states:  # The values of terminal states remain 0
                    self.V[state] = max(sum(prob * (R(state, action, next_state) + self.gamma * V_copy[next_state])
                                            for (prob, next_state) in T(state, action)) for action in self.mdp.actions)
                    delta = max(delta, abs(self.V[state] - V_copy[state]))

            i += 1
            print(f'Iteration {i}, delta: {delta:.6f}')
            if delta < self.epsilon:
                break

    def get_best_policy(self):
        """Return the policy induced by the calculated V table"""
        policy = {}
        for state in self.mdp.states:
            policy[state] = max(self.mdp.actions,
                                key=lambda action: self.expected_return(state, action))
        return policy

    def expected_return(self, state, action):
        """Compute the expected return of taking action a in state s"""
        return sum(prob * (self.mdp.R(state, action, next_state) + self.gamma * self.V[next_state])
                   for (prob, next_state) in self.mdp.T(state, action))






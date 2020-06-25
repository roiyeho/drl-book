def value_iteration(mdp, epsilon=0.01):
    """Solving an MDP by value iteration."""
    V1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma

    i = 0
    while True:
        V = V1.copy()
        delta = 0

        for s in mdp.states:
            if s not in mdp.terminals:  # The values of terminal states remain 0
                V1[s] = max(sum(p * (R(s, a, s1) + gamma * V[s1]) for (p, s1) in T(s, a))
                            for a in mdp.actions)
                delta = max(delta, abs(V1[s] - V[s]))

        i += 1
        print(f'Iteration {i}, delta: {delta:.6f}')
        if delta < epsilon:
            return V

def best_policy(mdp, V):
    """Given an MDP and a value function V, determine the best policy,
        as a mapping from state to action."""
    pi = {}
    for s in mdp.states:
        pi[s] = max(mdp.actions, key=lambda a: expected_return(s, a, mdp, V))
    return pi

def expected_return(s, a, mdp, V):
    """The expected return of taking action a in state s, according to the MDP and V."""
    return sum(p * (mdp.R(s, a, s1) + mdp.gamma * V[s1]) for (p, s1) in mdp.T(s, a))



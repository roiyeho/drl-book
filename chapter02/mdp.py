class MDP:
    """A Markov Decision Process, defined by a set of states (including an initial state and terminal
       states), a set of actions, a transition model, a reward function, and a discount factor.
       For the transition model P(s'|s, a), instead of storing a probability number for each
       state/action/state triplet, we have T(s, a) which returns a list of (p, s') pairs."""

    def __init__(self, states=None, initial_state=None, terminals=None,
                 actions=None, gamma=1.0):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        self.states = states
        self.initial_state = initial_state
        self.terminals = terminals
        self.actions = actions
        self.gamma = gamma

    def T(self, state, action):
        """Transition model. From a state and action, return a list of (probability, next-state) pairs."""
        raise NotImplementedError

    def R(self, state, action, next_state):
        """Return a numeric reward for this transition."""
        raise NotImplementedError

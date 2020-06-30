# Author: Roi Yehoshua
# Date: June 2020

from abc import ABC, abstractmethod

class MDP(ABC):
    """A Markov Decision Process, defined by a set of states, a set of
    actions, a transition model, and a reward function.
    Author: Roi Yehoshua
    Date: June 2020"""
    def __init__(self, states, actions, initial_state=None, terminal_states=None):
        """
        :param states: a list of states
        :param actions: a list of actions
        :param initial_state: optional, the initial state
        :param terminal_states: optional, a list of terminal states
        """
        self.states = states
        self.actions = actions
        self.initial_state = initial_state
        self.terminal_states = terminal_states

    @abstractmethod
    def T(self, state, action):
        """The transition function
        :param state: current state
        :param action: chosen action
        :return: a tuple of (probability, next state)
        """
        pass

    @abstractmethod
    def R(self, state, action, next_state):
        """The reward function
        :param state: current state
        :param action: chosen action
        :param next_state: next state
        :return: a numerical reward
        """
        pass

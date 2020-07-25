import pickle

class GridWorldPolicy():
    """Load the grid world policy from a file"""
    def __init__(self, filename):
        with open(filename, 'rb') as file:
            self.policy = pickle.load(file)

    def get_action(self, state):
        return self.policy[state]
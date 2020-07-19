import numpy as np

class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = np.empty(max_size, dtype=np.object)
        self.max_size = max_size
        self.current_index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.current_index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.current_index = (self.current_index + 1) % self.max_size

    def sample(self, batch_size):
        indices = np.random.randint(self.size, size=batch_size)
        return self.buffer[indices]

    def __len__(self):
        return self.size

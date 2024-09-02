from collections import namedtuple, deque
import random

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', "done"))

class ReplayMemory:
    """A cyclic buffer of bounded size that holds the transitions observed recently."""

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)


    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        """Randomly sample a batch of transitions"""
        return random.sample(self.memory, batch_size)


    def __len__(self):
        """Returns the current size of the memory"""
        return len(self.memory)
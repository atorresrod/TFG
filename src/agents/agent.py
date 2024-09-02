from abc import ABC, abstractmethod
from gymnasium import Env

class Agent(ABC):
    def __init__(self, env: Env, discount_factor: float, learning_rate: float):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate


    @abstractmethod
    def get_action(self, obs) -> int:
        """Returns the action to take given the observation."""
        pass


    @abstractmethod
    def train(self, num_steps: int) -> dict:
        """Trains the agent for the given number of steps"""
        pass
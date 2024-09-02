from .agent import Agent
from gymnasium import Env
from abc import ABC, abstractmethod
import numpy as np

class EpsilonDecayAgent(Agent, ABC):
    def __init__(
            self, 
            env: Env, 
            discount_factor: float, 
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float, 
            final_epsilon: float
        ):
        super().__init__(env, discount_factor, learning_rate)
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon


    def get_action(self, obs) -> int:
        """
        Returns the action to take given the observation.
        The best action is returned with probability (1 - epsilon)
        otherwise a random action is returned to ensure exploration
        """
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_greedy_action(obs)


    def decay_epsilon(self):
        """Decays epsilon using the epsilon_decay rate"""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
       
        
    @abstractmethod
    def get_greedy_action(self, obs) -> int:
        """Returns the greedy action for the given observation"""
        pass
        


    
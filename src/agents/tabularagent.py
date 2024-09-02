from .epsilon_decay_agent import EpsilonDecayAgent
from collections import defaultdict
import numpy as np
from gymnasium import Env

class TabularAgent(EpsilonDecayAgent):
    def __init__(
            self, 
            env: Env, 
            discount_factor: float, 
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float, 
            final_epsilon: float
        ):
        super().__init__(env, discount_factor, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)
        
        # state -> action q_values
        self.q_values = defaultdict(lambda : np.zeros(env.action_space.n, dtype=np.float64))
        
        
    def get_greedy_action(self, obs) -> int:
        """Returns the best action to take given the observation."""
        best_action = np.argmax(self.q_values[obs])
        greedy_actions = np.argwhere(self.q_values[obs] == self.q_values[obs][best_action]).flatten()
        
        if len(greedy_actions) == 0:
            if np.isnan(self.q_values[obs][best_action]):
                return self.env.action_space.sample()
            
            return best_action
        
        return np.random.choice(greedy_actions)
    

    def get_policy(self) -> dict:
        """Returns the greedy policy learned from the q_values"""
        p = defaultdict(int)

        for state, actions in self.q_values.items():
            p[state] = np.argmax(actions)

        return p
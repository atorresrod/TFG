from agents.epsilon_decay_agent import EpsilonDecayAgent
import numpy as np
from typing import Callable

class AproxAgent(EpsilonDecayAgent):
    def __init__(
            self, 
            env, 
            discount_factor: float, 
            learning_rate: float, 
            initial_epsilon: float, 
            epsilon_decay: float, 
            final_epsilon: float,
            initial_weights: np.ndarray,
            aprox_fun: Callable,
            grad_fun: Callable
        ):
        super().__init__(env, discount_factor, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)
        
        self.aprox_fun = aprox_fun
        self.grad_fun = grad_fun
        self.weights = initial_weights
        
        
    def get_greedy_action(self, obs) -> int:
        """Returns the greedy action for a given observation using the current weights"""
        q_vals = np.empty((self.env.action_space.n,))
        for a in range(self.env.action_space.n):
            q_vals[a] = self.aprox_fun(obs, a, self.weights)

        return np.random.choice(np.flatnonzero(q_vals == q_vals.max()))

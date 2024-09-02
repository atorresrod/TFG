from gymnasium import Env
import numpy as np
from tqdm import tqdm
from agents.aproxagent import AproxAgent
import numpy as np
from typing import Callable

class LinearSemiGradientSarsa(AproxAgent):
    def __init__(
            self, 
            env: Env,
            discount_factor: float,
            learning_rate: float,
            initial_epsilon: float, 
            epsilon_decay: float, 
            final_epsilon: float, 
            initial_weights: np.ndarray,
            aprox_fun: Callable,
            grad_fun: Callable,
        ):
        super().__init__(
            env, 
            discount_factor, 
            learning_rate, 
            initial_epsilon, 
            epsilon_decay, 
            final_epsilon, 
            initial_weights, 
            aprox_fun, 
            grad_fun
        )
    
    def update(self, state, action, reward, next_state, next_action, done):
        """Updates the agent's q_values using the episode results"""
        q_val = self.aprox_fun(state, action, self.weights)

        if done:
            self.weights = (
                self.weights + self.learning_rate * (reward - q_val) * 
                self.grad_fun(state, action, self.weights)
            )
        else:
            q_val_next = self.aprox_fun(next_state, next_action, self.weights)
            target = reward + self.discount_factor * q_val_next - q_val
            self.weights = (
                self.weights + self.learning_rate * target * 
                self.grad_fun(state, action, self.weights)
            )


    def train(self, num_steps: int) -> dict:
        """Trains a Semi-Gradient Sarsa agent for the given number of steps."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()
        
        action = self.get_action(obs)

        for step in range(num_steps):
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_action = self.get_action(next_obs)
            done = terminated or truncated
            self.update(obs, action, reward, next_obs, next_action, done)
            obs = next_obs
            action = next_action
            self.decay_epsilon()

            if done:
                rewards.append(info["episode"]["r"].item())
                episode_lengths.append(info["episode"]["l"].item())
                episode_time.append(info["episode"]["t"].item())
                obs, info = self.env.reset()

        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time

        return results
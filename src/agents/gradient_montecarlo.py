from gymnasium import Env
import numpy as np
from tqdm import tqdm
from agents.aproxagent import AproxAgent
import numpy as np
from typing import Callable

class GradientMonteCarlo(AproxAgent):
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
            aprox_fun, grad_fun
        )

    
    def update(self, episode_results : list[tuple[int | tuple, int, float]]):
        """Updates the agent's q_values using the episode results"""

        for i in range(len(episode_results)):
            g = sum([x[2] * (self.discount_factor ** j) for j, x in enumerate(episode_results[i:])])
            state, action, _ = episode_results[i]
            q_val = self.aprox_fun(state, action, self.weights)
            self.weights = (
                self.weights + self.learning_rate * (g - q_val) * 
                self.grad_fun(state, action, self.weights)
            )


    def train(self, num_steps: int) -> dict:
        """Trains a Gradient Monte Carlo agent for the given number of steps."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()
        episode_results = []

        for step in range(num_steps):
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_results.append((obs, action, reward))
            obs = next_obs
            self.decay_epsilon()

            if terminated or truncated:
                rewards.append(info["episode"]["r"].item())
                episode_lengths.append(info["episode"]["l"].item())
                episode_time.append(info["episode"]["t"].item())
                self.update(episode_results)
                obs, info = self.env.reset()
                episode_results.clear()
            
        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time

        return results
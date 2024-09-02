from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent

class Sarsa(TabularAgent):
    def __init__(
            self,
            env: Env,
            discount_factor: float,
            learning_rate: float,
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float,
        ) -> None:
        """Initializes the agent."""
        super().__init__(env, discount_factor, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)


    def update(
            self, 
            state: int | tuple, 
            action: int, 
            reward: float, 
            next_state: int | tuple, 
            next_action: int) -> None:
        """Updates the agent's q_values using the episode results"""
        q_val = self.q_values[state][action]
        next_q_val = self.q_values[next_state][next_action]

        self.q_values[state][action] = (
            q_val + self.learning_rate * (reward + self.discount_factor * next_q_val - q_val)
        )
    

    def train(self, num_steps: int) -> dict:
        """Trains a Sarsa agent for the given number of steps."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()

        action = self.get_action(obs)

        for step in range(num_steps):
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_action = self.get_action(next_obs)
            self.update(obs, action, reward, next_obs, next_action)
            obs = next_obs
            action = next_action
            self.decay_epsilon()

            if terminated or truncated:
                rewards.append(info["episode"]["r"].item())
                episode_lengths.append(info["episode"]["l"].item())
                episode_time.append(info["episode"]["t"].item())
                obs, info = self.env.reset()
        
        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time

        return results

        
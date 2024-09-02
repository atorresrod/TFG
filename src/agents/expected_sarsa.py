from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent
from typing import Callable

class ExpectedSarsa(TabularAgent):
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
            next_state: int | tuple) -> None:
        """Updates the agent's q_values using the episode results"""
        q_val = self.q_values[state][action]

        q_estimate = 0.0
        best_action = self.get_greedy_action(next_state)

        for a in range(self.env.action_space.n):
            if a == best_action:
                q_estimate += self.discount_factor * (1 - self.epsilon + (self.epsilon / self.env.action_space.n)) * self.q_values[next_state][a]
            else:
                q_estimate += self.discount_factor * (self.epsilon / self.env.action_space.n) * self.q_values[next_state][a]


        self.q_values[state][action] = (
            q_val + self.learning_rate * (reward + q_estimate - q_val)
        )
    

    def train(self, num_steps: int) -> dict:
        """Trains an Expected Sarsa agent for the given number of episodes."""

        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()

        for step in range(num_steps):
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.update(obs, action, reward, next_obs)
            obs = next_obs
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
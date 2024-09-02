from collections import defaultdict
from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent

class MonteCarlo(TabularAgent):
    def __init__(
            self,
            env: Env,
            discount_factor: float,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            first_visit: bool
        ) -> None:
        """Initializes the agent."""
        super().__init__(env, discount_factor, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)
        
        # state -> number of times visited
        self.visits = defaultdict(int)
        self.first_visit = first_visit
         
                
    def update(self, episode_results : list[tuple[int | tuple, int, float]], episode_visits: defaultdict) -> None:
        """Updates the agent's q_values using the episode results"""
        g = 0

        # Loop using index in reverse order
        for i in reversed(range(len(episode_results))):
            state, action, reward = episode_results[i]
            g = self.discount_factor * g + reward

            if self.first_visit:
                episode_visits[(state, action)] -= 1
                if episode_visits[(state, action)] > 0:
                    continue

            self.visits[(state, action)] += 1
            num_visits = self.visits[(state, action)]
            q_val = self.q_values[state][action]
            self.q_values[state][action] = q_val + (1 / num_visits) * (g - q_val)


    def train(self, num_steps: int) -> dict:
        """Trains a Monte Carlo agent for the given number of episodes."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()
        episode_results = []
        episode_visits = defaultdict(int)

        for step in range(num_steps):
            action = self.get_action(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            episode_results.append((obs, action, reward))
            episode_visits[(obs, action)] += 1
            obs = next_obs
            self.decay_epsilon()

            if terminated or truncated:
                test = info["episode"]["r"]
                rewards.append(info["episode"]["r"].item())
                episode_lengths.append(info["episode"]["l"].item())
                episode_time.append(info["episode"]["t"].item())
                self.update(episode_results, episode_visits)
                episode_results.clear()
                episode_visits.clear()
                obs, info = self.env.reset()
        
        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time

        return results
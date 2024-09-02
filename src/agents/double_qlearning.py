from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent
from collections import defaultdict

class DoubleQLearning(TabularAgent):
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
        self.q_values_bias = defaultdict(lambda: np.zeros(env.action_space.n))


    def get_greedy_action(self, obs) -> int:
        """
        Returns the best action to take given the observation.
        """
        # Calculate Q1 + Q2
        q_values_sum = self.q_values[obs] + self.q_values_bias[obs]
        # Choose action with highest Q1 + Q2 value
        best_action = np.argmax(q_values_sum)
        greedy_actions = np.argwhere(q_values_sum == q_values_sum[best_action]).flatten()
        return np.random.choice(greedy_actions)


    def update(
            self, 
            state: int | tuple, 
            action: int, 
            reward: float, 
            next_state: int | tuple) -> None:
        """Updates the agent's q_values using the episode results"""
        
        if (np.random.random() < 0.5):
            # Update Q1
            self.q_values[state][action] += (
                self.learning_rate * (reward + self.discount_factor * \
                                      self.q_values_bias[next_state][np.argmax(self.q_values[next_state])] \
                                       - self.q_values[state][action])
            )
        else:
            # Update Q2
            self.q_values_bias[state][action] += (
                self.learning_rate * (reward + self.discount_factor * \
                                      self.q_values[next_state][np.argmax(self.q_values_bias[next_state])] \
                                       - self.q_values_bias[state][action])
            )

    
    def train(self, num_steps: int) -> dict:
        """Trains a Double Q-Learning agent for the given number of steps."""
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

        
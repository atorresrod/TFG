from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent

class NStepSarsa(TabularAgent):
    def __init__(
            self, 
            env: Env,
            discount_factor: float,
            learning_rate: float,
            initial_epsilon: float, 
            epsilon_decay: float,
            final_epsilon: float,
            step: int,
        ) -> None:
        """Initializes the agent."""
        super().__init__(env, discount_factor, learning_rate, initial_epsilon, epsilon_decay, final_epsilon)
        self.step = step


    def update(
            self,
            current_update_step: int,
            episode_results: list[tuple[int | tuple, int, float]],
            next_state: int | tuple,
            next_action: int,
            final_time_step: int
    ):
        """Updates the agent's q_values using the episode results"""
        g = 0
        for i in range(current_update_step, min(current_update_step + self.step, final_time_step)):
            _, _, reward = episode_results[i]
            g += (self.discount_factor ** (i - current_update_step)) * reward

        if current_update_step + self.step < final_time_step:
            g += (self.discount_factor ** self.step) * self.q_values[next_state][next_action]

        state, action, _ = episode_results[current_update_step]
        q_val = self.q_values[state][action]
        self.q_values[state][action] = q_val + self.learning_rate * (g - q_val)
    

    def train(self, num_steps: int) -> dict:
        """Trains a n-step Sarsa agent for the given number of episodes."""

        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()
        done = False
        action = self.get_action(obs)
        timestep = 0
        episode_results = []
        current_update_step = -1
        final_time_step = np.inf
        next_obs = 0
        next_action = 0

        for _ in range(num_steps):
            
            if current_update_step < final_time_step - 1:
                if timestep < final_time_step:
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    episode_results.append((obs, action, reward))

                    if done:
                        final_time_step = timestep + 1
                    else:
                        next_action = self.get_action(next_obs)

                current_update_step = timestep - self.step + 1

                if current_update_step >= 0:
                    self.update(current_update_step, episode_results, next_obs, next_action, final_time_step)

                obs = next_obs
                action = next_action
                timestep += 1

                self.decay_epsilon()
            else:
                rewards.append(info["episode"]["r"].item())
                episode_lengths.append(info["episode"]["l"].item())
                episode_time.append(info["episode"]["t"].item())
                obs, info = self.env.reset()
                done = False
                action = self.get_action(obs)
                timestep = 0
                episode_results = []
                current_update_step = -1
                final_time_step = np.inf
                next_obs = 0
                next_action = 0

        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time
        results["policy"] = self.get_policy()

        return results

        
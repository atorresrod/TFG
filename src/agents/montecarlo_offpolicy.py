from collections import defaultdict
from gymnasium import Env
import numpy as np
from tqdm import tqdm
from .tabularagent import TabularAgent

class MonteCarloOffPolicy(TabularAgent):
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

        self.cum_sum_weights = defaultdict(int)
        self.behaviour_policy = defaultdict(lambda: self.env.action_space.sample())


    def initialize_behaviour_policy(self) -> None:
        """Initializes the behaviour policy to a epsilon greedy policy respect
        to the q values"""
        for state in range(self.env.observation_space.n):
            self.behaviour_policy[state] = np.zeros(self.env.action_space.n)
            self.behaviour_policy[state].fill(self.epsilon / self.env.action_space.n)

            best_action = (
                np.random.choice(
                    np.flatnonzero(self.q_values[state] == self.q_values[state].max())
                )
            )
            self.behaviour_policy[state][best_action] = (
                1 - self.epsilon + self.epsilon / self.env.action_space.n
            )


    def get_action(self, obs) -> int:
        """
        Returns the action to take given the observation using the behaviour policy
        """
        return np.random.choice(
            np.arange(self.env.action_space.n),
            p=self.behaviour_policy[obs]
        )


    def update(self, episode_results : list[tuple[int | tuple, int, float]]) -> None:
        """Updates the agent's q_values using the episode results"""
        g = 0
        weight = 1

        # Loop using index in reverse order
        for i in reversed(range(len(episode_results))):
            state, action, reward = episode_results[i]
            g = self.discount_factor * g + reward
            self.cum_sum_weights[(state, action)] += weight

            q_val = self.q_values[state][action]
            self.q_values[state][action] = (
                q_val + (weight / self.cum_sum_weights[(state, action)]) *
                (g - q_val)
            )

            best_action = np.argmax(self.q_values[state])
            if action != best_action:
                break

            weight = weight / self.behaviour_policy[state][action]


    def train(self, num_steps: int) -> dict:
        """Trains a Monte Carlo Off-Policy agent for the given number of steps."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        obs, info = self.env.reset()
        episode_results = []
        self.initialize_behaviour_policy()

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
                episode_results.clear()
                self.initialize_behaviour_policy()
                

                obs, info = self.env.reset()

        results["rewards"] = rewards
        results["episode_lengths"] = episode_lengths
        results["episode_time"] = episode_time

        return results


import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .dqn import DQN
from ..epsilon_decay_agent import EpsilonDecayAgent
from .replay_memory import ReplayMemory, Transition


class DQNAgent(EpsilonDecayAgent):
    """Implements a DQN agent."""
    def __init__(
            self,
            env: gym.Env,
            initial_epsilon: float,
            epsilon_decay_end_frame: float,
            final_epsilon: float,
            lr: float,
            discount_factor: float,
            target_update_frequency: int,
            learning_start_step: int,
            replay_memory_size: int,
            batch_size: int,
            save_progress_path: str = None,
            save_progress_episodes: int = 100
    ):
        """Initializes de agent."""
        super().__init__(env, discount_factor, lr, initial_epsilon, (initial_epsilon-final_epsilon)/(epsilon_decay_end_frame - learning_start_step), final_epsilon)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_update_frequency = target_update_frequency
        self.learning_start_step = learning_start_step
        self.batch_size = batch_size
        self.replay_memory = ReplayMemory(replay_memory_size)
        self.policy_net = DQN(env.action_space.n).to(self.device)
        self.target_net = DQN(env.action_space.n).to(self.device)
        self.update_target_net_weights()
        self.target_net.eval()
        self.optimizer = optim.RMSprop(
            self.policy_net.parameters(),
            lr=lr,
            alpha=0.95,
            eps = 0.01,
        )
        self.loss_fn = nn.HuberLoss()
        self.save_progress_path = save_progress_path
        self.save_progress_episodes = save_progress_episodes

    def update_target_net_weights(self):
        """Updates the target network weights with the behaviour network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_greedy_action(self, obs) -> int:
        """Returns the best action to take given the observation."""
        state = torch.from_numpy(np.array(obs)).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state)
            best_action = q_values.max(1).indices.view(1,1)
            return best_action.item()
        
    def optimize_model(self) -> None:
        """Optimizes the model using a batch of transitions."""
        if len(self.replay_memory) < self.batch_size:
            return
        
        transitions = self.replay_memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        actions = torch.from_numpy(np.array(batch.action)).long().to(self.device)
        rewards = torch.from_numpy(np.array(batch.reward)).float().to(self.device)
        next_states = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        dones = torch.from_numpy(np.array(batch.done)).int().to(self.device)

        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1).values
            target_q_values = rewards + (1 - dones) * self.discount_factor * next_q_values

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_steps: int) -> dict:
        """Trains a DQN agent for the given number of steps."""
        results = dict()
        rewards = []
        episode_lengths = []
        episode_time = []

        state, info = self.env.reset()

        episode_counter = 0
        ep_reward = []

        for step in range(num_steps):
            action = self.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_memory.push(state, action, next_state, reward, int(terminated))

            state = next_state

            if step > self.learning_start_step:
                self.optimize_model()

            if step % self.target_update_frequency == 0:
                self.update_target_net_weights()

            if done:
                episode_counter += 1
                ep_reward.append(info['episode']['r'].item())

                rewards.append(info['episode']['r'].item())
                episode_lengths.append(info['episode']['l'].item())
                episode_time.append(info['episode']['t'].item())

                if self.save_progress_path and episode_counter % self.save_progress_episodes == 0:
                    # Save progress to file
                    mean_reward = np.mean(ep_reward)
                    file = open(self.save_progress_path, "a")
                    text = "Episode: " + str(episode_counter) + " Iter: " + str(step) + " Epsilon: " + str(self.epsilon) +" Reward: " + str(mean_reward) + "\n"
                    file.write(text)
                    file.close()
                    
                    ep_reward = []

                # Reset environment
                state, info = self.env.reset()

            if step > self.learning_start_step:
                self.decay_epsilon()

        results['rewards'] = rewards
        results['episode_lengths'] = episode_lengths
        results['episode_time'] = episode_time

        return results
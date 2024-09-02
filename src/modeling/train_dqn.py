"""
Trains a DQN agent using the same hyperparameters as the original DQN paper.
"""

from agents import DQNAgent
import gymnasium as gym
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="The name of the atari environment to train on")
    parser.add_argument("save_model_path", type=str, help="The path to save the model")
    parser.add_argument("save_metrics_path", type=str, help="The path to save the training metrics")
    parser.add_argument("--progress_path", type=str, help="The path to save the training progress")
    parser.add_argument("--save_progress_episodes", type=str, help="The number of episodes to save progress for", default=100)
    args = parser.parse_args()


    env = gym.make(args.env)
    env = gym.wrappers.AtariPreprocessing(
        env=env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = gym.wrappers.FrameStack(env, num_stack=4)
    env = gym.experimental.wrappers.ClipRewardV0(env, -1.0, 1.0)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    agent = DQNAgent(
        env=env,
        initial_epsilon=1.0,
        epsilon_decay_end_frame=1_000_000,
        final_epsilon=0.1,
        lr=0.00025,
        discount_factor=0.99,
        target_update_frequency=10_000,
        learning_start_step=50_000,
        replay_memory_size=1_000_000,
        batch_size=32,
        save_progress_path=args.progress_path if args.progress_path else None,
        save_progress_episodes=args.save_progress_episodes,
    )

    train_results = agent.train(50_000_000)

    torch.save(agent.policy_net.state_dict(), args.save_model_path)
    torch.save(train_results, args.save_metrics_path)
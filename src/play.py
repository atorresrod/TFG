"""
This python script is used to play the game using the trained model. The script takes in the environment name, the path to the
trained model, and the method to use (DQN or tabular). The script then loads the model and plays the game using the model.
As optional parameters the script can take the number of episodes to play and the epsilon value for the epsilon-greedy policy.
"""
import gymnasium as gym
from agents.dqn.dqn import DQN
import argparse
import torch
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="The name of the atari environment to play on")
    parser.add_argument("model_path", type=str, help="The path to trained the model")
    parser.add_argument("is_dqn_method", type=int, help="Set to 1 to use DQN method, 0 for tabular")
    parser.add_argument("--num_episodes", type=int, help="The number of episodes to play", default=1)
    parser.add_argument("--epsilon", type=float, help="The epsilon value for epsilon-greedy policy", default=0.00)
    args = parser.parse_args()

    # Create the environment
    if args.env == "FrozenLake-v1":
        env = gym.make(args.env, map_name="8x8", render_mode="human")
    else:
        env = gym.make(args.env, render_mode="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dqn_method:
        # Add the Atari preprocessing
        env = gym.wrappers.AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            grayscale_newaxis=False,
            scale_obs=True
        )
        env = gym.wrappers.FrameStack(env, num_stack=4)

        # Load the DQN model and set it to evaluation mode
        model = DQN(env.action_space.n).to(device)
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    else:
        # Load the tabular policy
        model = torch.load(args.model_path)

    for _ in range(args.num_episodes):
        obs, info = env.reset()
        done = False

        # Run the game for 1 episode
        while not done:
            # epsilon-greedy action selection
            if np.random.random() < args.epsilon:
                action = env.action_space.sample()
            elif args.dqn_method:
                obs = torch.from_numpy(np.array(obs)).float().to(device).unsqueeze(0)
                action = model(obs).argmax(1).item()
            else:
                action = model[obs]

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
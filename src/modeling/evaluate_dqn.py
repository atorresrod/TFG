from agents import DQN
import gymnasium as gym
import torch
import numpy as np
import random
import pandas as pd

def evaluate(env: gym.Env, model: DQN, num_episodes: int, epsilon: float):
    """Evaluates a DQN agent for the given number of episodes in the given environment using
    a specific epsilon value for the epsilon-greedy policy."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.from_numpy(np.array(obs)).float().to(device).unsqueeze(0)
        done = False
        total_reward = 0

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = model(obs).argmax(1).item()

            obs, reward, done, _ , _ = env.step(action)
            obs = torch.from_numpy(np.array(obs)).float().to(device).unsqueeze(0)
            total_reward += reward

        rewards.append(total_reward)

    env.close()

    avg = np.average(rewards)
    std = np.std(rewards)
    max = np.max(rewards)

    print(f"Environment: {env.unwrapped.spec.id}, Average Reward: {avg}, Standard Deviation: {std}, Max Reward: {max}")
    print(rewards)

    return avg, std, max

if __name__ == "__main__":
    NUM_EPISODES = 30
    EPSILON = 0.05

    results = {
        "Environment": [],
        "Average Reward": [],
        "Standard Deviation": [],
        "Max Reward": []
    }


    # Evaluate the agents in the Breakout environment
    env = gym.make("BreakoutNoFrameskip-v4")
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

    model = DQN(env.action_space.n).to("cuda")
    model.load_state_dict(torch.load("saves/models/breakout/dqn.pth"))
    model.eval()

    avg, std, mx = evaluate(env, model, NUM_EPISODES, EPSILON)

    results["Environment"].append("BreakoutNoFrameskip-v4")
    results["Average Reward"].append(avg)
    results["Standard Deviation"].append(std)
    results["Max Reward"].append(mx)

    # Evaluate the agents in the Space Invaders environment
    env = gym.make("SpaceInvadersNoFrameskip-v4")
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

    model = DQN(env.action_space.n).to("cuda")
    model.load_state_dict(torch.load("saves/models/space_invaders/dqn.pth"))
    model.eval()

    avg, std, mx = evaluate(env, model, NUM_EPISODES, EPSILON)

    results["Environment"].append("SpaceInvadersNoFrameskip-v4")
    results["Average Reward"].append(avg)
    results["Standard Deviation"].append(std)
    results["Max Reward"].append(mx)
    
    # Evaluate the agents in the Enduro environment
    env = gym.make("EnduroNoFrameskip-v4")
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

    model = DQN(env.action_space.n).to("cuda")
    model.load_state_dict(torch.load("saves/models/enduro/dqn.pth"))
    model.eval()

    avg, std, mx = evaluate(env, model, NUM_EPISODES, EPSILON)

    results["Environment"].append("EnduroNoFrameskip-v4")
    results["Average Reward"].append(avg)
    results["Standard Deviation"].append(std)
    results["Max Reward"].append(mx)

    # Evaluate the agents in the VideoPinball environment
    env = gym.make("VideoPinballNoFrameskip-v4")
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
    
    model = DQN(env.action_space.n).to("cuda")
    model.load_state_dict(torch.load("saves/models/pinball/dqn.pth"))

    avg, std, mx = evaluate(env, model, NUM_EPISODES, EPSILON)

    results["Environment"].append("VideoPinballNoFrameskip-v4")
    results["Average Reward"].append(avg)
    results["Standard Deviation"].append(std)
    results["Max Reward"].append(mx)

    print("----------- Final Results -----------")
    print(results)
    print ("------------------------------------")
    df = pd.DataFrame(results)
    print(df)



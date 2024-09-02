import gymnasium as gym
import torch
import numpy as np
from collections import defaultdict
import pandas as pd

def evaluate(env: gym.Env, policy: dict | defaultdict, num_episodes: int):
    """Evaluates a tabular agent for the given number of episodes in the given environment."""
    rewards = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = policy[obs]
            obs, reward, terminated, truncated , _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1

        rewards.append(total_reward)

    env.close()

    avg = np.average(rewards)
    std = np.std(rewards)
    mx = np.max(rewards)

    return avg, std, mx


if __name__ == "__main__":
    NUM_EPISODES = 30

    results = {
        "Environment": [],
        "Algorithm": [],
        "Average Reward": [],
        "Standard Deviation": [],
        "Max Reward": [],
    }

    algorithms = [
        "MonteCarloEveryVisit", 
        "MonteCarloFirstVisit", 
        "MonteCarloOffPolicy",
        "Sarsa",
        "QLearning",
        "ExpectedSarsa",
        "DoubleQLearning",
        "NStepSarsa",
    ]

    # Evaluate the agents in the FrozenLake environment
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    for alg in algorithms:
        policy = torch.load(f"saves/models/frozen_lake/FrozenLake-v1-{alg}.pth")
        avg, std, mx = evaluate(env, policy, NUM_EPISODES)

        results["Environment"].append("FrozenLake-v1")
        results["Algorithm"].append(alg)
        results["Average Reward"].append(avg)
        results["Standard Deviation"].append(std)
        results["Max Reward"].append(mx)

        print(f"Environment: FrozenLake-v1, Algorithm: {alg}, Average Reward: {avg}, Standard Deviation: {std}")
    
    # Evaluate the agents in the CliffWalking environment
    env = gym.make("CliffWalking-v0")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    for alg in algorithms:
        policy = torch.load(f"saves/models/cliffwalking/CliffWalking-v0-{alg}.pth")
        avg, std, mx = evaluate(env, policy, NUM_EPISODES)

        results["Environment"].append("CliffWalking-v0")
        results["Algorithm"].append(alg)
        results["Average Reward"].append(avg)
        results["Standard Deviation"].append(std)
        results["Max Reward"].append(mx)

        print(f"Environment: CliffWalking-v0, Algorithm: {alg}, Average Reward: {avg}, Standard Deviation: {std}")

    # Evaluate the agents in the Taxi environment
    env = gym.make("Taxi-v3")
    for alg in algorithms:
        policy = torch.load(f"saves/models/taxi/Taxi-v3-{alg}.pth")
        avg, std, mx = evaluate(env, policy, NUM_EPISODES)

        results["Environment"].append("Taxi-v3")
        results["Algorithm"].append(alg)
        results["Average Reward"].append(avg)
        results["Standard Deviation"].append(std)
        results["Max Reward"].append(mx)

        print(f"Environment: Taxi-v3, Algorithm: {alg}, Average Reward: {avg}, Standard Deviation: {std}")

    # Print the results
    print("--------------------------------")
    print(results)


    print("--------------------------------")
    df = pd.DataFrame(results)
    pd.set_option('display.max_columns', None)
    print(df)
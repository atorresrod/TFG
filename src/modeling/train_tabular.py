"""
Trains a variety of tabular agents on the given environment and saves the trained models and metrics.
"""

from agents import *
import gymnasium as gym
import torch
import argparse

def save_model(save_model_dir: str, save_metrics_dir: str, env_name: str, algorithm_name: str, model, metrics):
    """Saves the trained model and metrics to the given directories."""
    save_model_path = save_model_dir + "/" + env_name + "-" + algorithm_name + ".pth"
    save_metrics_path = save_metrics_dir + "/" + env_name + "-" + algorithm_name + ".pth"
    torch.save(model, save_model_path)
    torch.save(metrics, save_metrics_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="The name of the gym environment to train on")
    parser.add_argument("save_model_dir_path", type=str, help="The directory path to save the trained models")
    parser.add_argument("save_metrics_dir_path", type=str, help="The directory path to save the training metrics")
    args = parser.parse_args()

    if args.env == "FrozenLake-v1":
        env = gym.make(args.env, map_name="8x8", is_slippery=True)
    else:
        env = gym.make(args.env)

    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Hyperparameterss
    LR = 0.01
    DISCOUNT_FACTOR = 0.99
    INITIAL_EPSILON = 1.0
    FINAL_EPSILON = 0.1
    NUM_STEPS = 50_000_000
    EPSILON_DECAY = INITIAL_EPSILON / (NUM_STEPS / 5)

    agent = MonteCarlo(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        first_visit=True
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__ + "FirstVisit", 
        agent.get_policy(), 
        results
    )

    agent = MonteCarlo(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        first_visit=False
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__ + "EveryVisit", 
        agent.get_policy(),
        results
    )

    agent = MonteCarloOffPolicy(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(),
        results
    )

    agent = Sarsa(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(), 
        results
    )

    agent = QLearning(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(), 
        results
    )

    agent = DoubleQLearning(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(), 
        results
    )

    agent = ExpectedSarsa(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(), 
        results
    )

    agent = NStepSarsa(
        env=env,
        discount_factor=DISCOUNT_FACTOR,
        learning_rate=LR,
        initial_epsilon=INITIAL_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        final_epsilon=FINAL_EPSILON,
        step=5
    )

    results = agent.train(NUM_STEPS)
    save_model(
        args.save_model_dir_path, 
        args.save_metrics_dir_path, 
        args.env, 
        agent.__class__.__name__, 
        agent.get_policy(), 
        results
    )

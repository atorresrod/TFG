import agents
import gymnasium as gym

def evaluate_agent(env: gym.Env, agent: agents.TabularAgent):
    """Evaluates a tabular agent for 10 episodes and returns the average reward."""
    # Get the greedy policy learnt by the agent
    policy = agent.get_policy()

    cummulative_reward = 0

    # Run 10 episodes using the policy learnt by the agent
    for i in range(10):
        # Reset the environment
        obs, info = env.reset()
        done = False

        while not done:
            action = policy[obs]
            obs, reward, terminated, truncated, info = env.step(action)
            cummulative_reward += reward
            done = terminated or truncated

    env.close()

    # Calculate the average over the 10 episodes
    cummulative_reward /= 10

    return cummulative_reward


def train_agent(env_name, algorithm, lr, discount_factor, num_steps):
    """Trains an agent using the given hyperparameters and returns the trained agent and the environment."""
    initial_epsilon = 1.0
    epsilon_decay = initial_epsilon / (num_steps / 2)
    final_epsilon = 0.1

    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = None

    match algorithm:
        case agents.MonteCarlo:
            agent = agents.MonteCarlo(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                first_visit=True,
            )
        case agents.MonteCarloOffPolicy:
            agent = agents.MonteCarloOffPolicy(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
            )
        case agents.Sarsa:
            agent = agents.Sarsa(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
            )
        case agents.ExpectedSarsa:
            agent = agents.ExpectedSarsa(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon
            )
        case agents.QLearning:
            agent = agents.QLearning(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon
            )
        case agents.DoubleQLearning:
            agent = agents.DoubleQLearning(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon
            )
        case agents.NStepSarsa:
            agent = agents.NStepSarsa(
                env=env,
                discount_factor=discount_factor,
                learning_rate=lr,
                initial_epsilon=initial_epsilon,
                epsilon_decay=epsilon_decay,
                final_epsilon=final_epsilon,
                step=5
            )
        case _:
            raise("Invalid algorithm class")
        
    # Train the agent
    agent.train(num_steps)

    # Return the trained agent and the environment
    return agent, env

def get_best_set_of_hyperparameters(results):
    """Returns the best set of hyperparameters and the corresponding reward."""
    average_reward = {}

    for res in results:
        key = (res["lr"], res["discount_factor"], res["num_steps"])

        if key not in average_reward:
            average_reward[key] = []

        average_reward[key].append(res["reward"])

    best_set_of_hyperparameters = None
    best_reward = -float("inf")
    
    for key, rewards in average_reward.items():
        avg = sum(rewards) / len(rewards)

        if avg > best_reward:
            best_reward = avg
            best_set_of_hyperparameters = key

    return best_set_of_hyperparameters, best_reward

if __name__ == "__main__":
    envs = ["CliffWalking-v0"]
    algorithms = [
        agents.MonteCarlo, 
        agents.Sarsa,
    ]

    hyperparameters = {
        "lr": [0.001, 0.01, 0.1],
        "discount_factor": [0.99],
        "num_steps": [20_000_000],
    }

    results = []

    for env_name in envs:
        for algorithm in algorithms:
            for lr in hyperparameters["lr"]:
                for discount_factor in hyperparameters["discount_factor"]:
                    for num_steps in hyperparameters["num_steps"]:
                        agent, env = train_agent(env_name, algorithm, lr, discount_factor, num_steps)
                        reward = evaluate_agent(env, agent)
                        results.append({
                            'env': env_name,
                            'algorithm': algorithm,
                            'lr': lr,
                            'discount_factor': discount_factor,
                            'reward': reward,
                            'num_steps': num_steps
                        })
                        print(results[-1])

    best_hyperparams, best_reward = get_best_set_of_hyperparameters(results)

    print("---------------------------------")
    print(f"Best set of hyperparameters: {best_hyperparams} with reward {best_reward}")
import matplotlib.pyplot as plt

from pathlib import Path
import os
import sys
path = Path(os.getcwd())

sys.path.append(str(path))
print("\n>>>", str(path), "\n")


from engines.environments import BlackScholesEnv
from engines.agents.models import PolicyNN
from dto.input_dtos import EnvParams
from engines.agents.actor_behaviour_cloning import BehaviorReplicationAgent
from dto.config_loader import load_config
from engines.gpu_manager import setup_gpu_optimizations

def debug_delta_hedge_trajectory():
    """
    Debug the delta hedge rollouts using BhaviorReplicationAgent and plot the results.
    """
    # Define environment parameters
    config_path = "runs/hyperparameters.yml"
    params_key = "v_debug"
    env_params, algo_params, _ = load_config(path=config_path, version=params_key)

    # Initialize the Black-Scholes environment
    env = BlackScholesEnv(env_params)

    # Create a simple policy network structure for behavior cloning
    policy_net = PolicyNN(algo_params)

    # Initialize the BehaviorReplicationAgent
    agent = BehaviorReplicationAgent(
        env=env,
        policy_net=policy_net,
        hyperparameters_version=params_key,
        root_repo=str(path)
    )

    # Perform rollouts
    results = agent.simulate_delta_hedge_trajectory(batch_size=10_000)

    # Plot the results
    stock_prices = results[:,:,0]  # Replace with actual keys from the returned data
    time_steps = results[:,:,4] # Replace with actual keys from the returned data
    hedge_positions = results[:,:,5]  # Replace with actual keys from the returned data

    plt.figure(figsize=(10, 8))

    # Create a scatter plot with stock prices on x-axis, time to maturity on y-axis, and hedge positions as color
    scatter = plt.scatter(
        stock_prices.flatten(), 
        time_steps.flatten(), 
        c=hedge_positions.flatten(), 
        cmap="viridis", 
        s=10, 
        alpha=0.8
    )

    plt.colorbar(scatter, label="Hedge Position")
    plt.xlabel("Stock Price")
    plt.ylabel("Time to Maturity")
    plt.title("Hedge Position by Stock Price and Time to Maturity")
    plt.grid(alpha=0.5)

    plt.tight_layout()
    plt.show()


def debug_BC_train():
    """
    Debug the training of the BehaviorReplicationAgent.
    """
    # Define environment parameters
    config_path = "runs/hyperparameters.yml"
    params_key = "v_debug"
    env_params, algo_params, _ = load_config(path=config_path, version=params_key)

    # Initialize the Black-Scholes environment
    env = BlackScholesEnv(env_params)

    # Create a simple policy network structure for behavior cloning
    policy_net = PolicyNN(algo_params, env)

    # Initialize the BehaviorReplicationAgent
    agent = BehaviorReplicationAgent(
        env=env,
        policy_net=policy_net,
        hyperparameters_version=params_key,
        root_repo=str(path)
    )

    # Train the agent using expert trajectories
    agent.train_behavior_cloning(
        epochs=10_000, 
        lr=1e-2, 
        lambda_entropy=0.01,
        batch_size=agent.batch_size)
    
if __name__ == "__main__":

    # debug_delta_hedge_trajectory()
    setup_gpu_optimizations()
    debug_BC_train()
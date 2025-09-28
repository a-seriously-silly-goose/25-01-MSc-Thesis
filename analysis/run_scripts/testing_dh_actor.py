from agents.actor_delta_hedge import DeltaHedgeActor
from envs import BlackScholesEnv
import torch as T
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


def setup_experiment(hyperparameters_version, is_training=False, preload=False):
    """
    Sets up the experimental configuration, retrieves hyperparameters, and initializes directories.
    """
    RUNS_DIR = "runs"
    os.makedirs(RUNS_DIR, exist_ok=True)
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    root_path = os.path.join(os.path.dirname(os.getcwd()))

    # Load hyperparameters
    with open(os.path.join(root_path, "hyperparameters.yml"), "r") as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameters_version]

    envParams = hyperparameters["envParams"]
    algoParams = hyperparameters["algoParams"]
    riskParams = hyperparameters["riskParams"]
    runParams = hyperparameters["runParams"]
    repo_name = hyperparameters_version

    # Log message
    log_message = (
        f"*** Name of the repository:  {repo_name} ***\n"
        f"*** Environment parameters:  {envParams} ***\n"
        f"*** Algorithm parameters:  {algoParams} ***\n"
        f"*** Risk measures parameters:  {riskParams} ***\n"
        f"*** Run parameters:  {runParams} ***\n"
    )
    print(log_message)

    # Assert Feller condition
    assert (
        2 * envParams["kappa"] * envParams["theta"] > envParams["eta"] ** 2
    ), "Feller condition is not satisfied."

    # Create repository
    repo = os.path.join(RUNS_DIR, repo_name)
    os.makedirs(repo, exist_ok=True)

    # Save log file
    LOG_FILE = os.path.join(repo, f"{hyperparameters_version}.log")
    with open(LOG_FILE, "w") as file:
        file.write(log_message + "\n")

    return {
        "device": device,
        "envParams": envParams,
        "algoParams": algoParams,
        "riskParams": riskParams,
        "runParams": runParams,
        "repo": repo,
        "log_file": LOG_FILE,
        "hyperparameters_version": hyperparameters_version,
        "is_training": is_training,
        "preload": preload,
    }


def plot_env_paths(env, n_paths=10, n_steps=100, save_path="stock_price_paths.png"):
    """
    Simulates stock price paths and plots them.

    Parameters:
        env (BlackScholesEnv): The environment to simulate.
        n_paths (int): Number of paths to simulate.
        n_steps (int): Number of time steps to simulate.
        save_path (str): Path to save the generated plot.
    """
    # Reset environment for initial conditions
    S, v, alpha, B = env.reset(Nsims=n_paths)

    # Initialize storage for paths
    stock_paths = T.zeros((n_steps, n_paths), dtype=T.float)
    stock_paths[0, :] = S  # Initial stock price

    # Simulate paths
    for t in range(1, n_steps):
        alpha_tm1 = alpha
        B_t = B

        # Simple assumption for no hedging changes (constant alpha)
        alpha_t = alpha_tm1

        # Perform environment step
        S, v, alpha, B, _ = env.step(S, v, alpha_tm1, B_t, alpha_t)
        stock_paths[t, :] = S  # Store current step's stock prices

    # Plot the paths
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(stock_paths[:, i], label=f"Path {i + 1}")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    # plt.legend(loc="best", ncol=2, fontsize="small", frameon=False)
    plt.tight_layout()

    # Save the plot to the specified path
    plt.savefig(save_path)
    print(f"Stock price paths plot saved at: {save_path}")
    plt.show()


# Load configuration and setup environment
config = setup_experiment(
    hyperparameters_version="d002.000.001", is_training=True, preload=False
)

env = BlackScholesEnv(config["envParams"])
DH_agent = DeltaHedgeActor(
    os.getcwd(),
    "DH_actor",
    env,
    config["hyperparameters_version"],
    config["log_file"],
)

# Simulate and plot environment paths
plot_env_paths(env, n_paths=1000, n_steps=100)

# Test existing DeltaHedgeActor plots
DH_agent.plot_current_policy()
DH_agent.plot_delta_vs_time()
DH_agent.plot_delta_vs_stock_price()

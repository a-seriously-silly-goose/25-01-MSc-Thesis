import matplotlib.pyplot as plt
import torch as T
import numpy as np
from envs import BlackScholesEnv


# Function to simulate paths and plot
def simulate_and_plot_paths(env, n_steps=100, n_paths=10):
    """
    Simulates stock price paths for the given environment and plots them.

    :param env: BlackScholesEnv instance
    :param n_steps: Number of time steps
    :param n_paths: Number of paths to simulate
    """
    # Reset environment for initial conditions
    S, v, alpha, B = env.reset(Nsims=n_paths)

    # Set up storage for paths
    all_paths = T.zeros((n_steps, n_paths), dtype=T.float)
    all_paths[0, :] = S  # Initial stock prices

    # Simulate paths
    for t in range(1, n_steps):
        alpha_tm1 = alpha
        B_t = B

        # No hedging or action for simplicity (alpha_t remains the same)
        alpha_t = alpha_tm1
        S_t, v_t = S, v

        # Calculate next state using the environment's step function
        S_tp1, v_tp1, alpha_tp1, B_tp1, _ = env.step(S_t, v_t, alpha_tm1, B_t, alpha_t)

        # Store the stock price for the step
        all_paths[t, :] = S_tp1

        # Update the state for the next iteration
        S, v, alpha, B = S_tp1, v_tp1, alpha_tp1, B_tp1

    # Plot the paths
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(all_paths[:, i].numpy(), label=f"Path {i + 1}")
    plt.xlabel("Time Step")
    plt.ylabel("Stock Price")
    plt.title("Simulated Stock Price Paths")
    # plt.legend(loc="best", ncol=2, fontsize="small", frameon=False)
    plt.tight_layout()

    # Save the plot
    plot_path = "stock_price_paths.png"
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()


# Set up Black-Scholes environment
env_params = {
    "T": 1.0,  # Maturity, 1 year
    "Ndt": 100,  # Number of time steps
    "K": 10,  # Strike price
    "r": 0.01,  # Risk-free rate
    "sigma": 0.02,  # Volatility
    "v0": 0.02,
    "epsilon": 0.01,  # Transaction cost
    "S0": 10,  # Initial stock price
    "B0": 0,  # Initial bank account balance
    "mu": 0.01,  # drift in GBM
    "epsilon": 0,  # transaction cost
    "alpha0": 0,  # initial position
    "max_alpha": 8,  # maximum position
    "Ndt": 31,  # number of time steps
    "kappa": 9,
    "theta": 0.1,
    "eta": 0.1,
}
env = BlackScholesEnv(env_params)

# Simulate stock price paths and plot
simulate_and_plot_paths(env, n_steps=100, n_paths=1000)

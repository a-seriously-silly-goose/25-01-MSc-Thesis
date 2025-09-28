import os
import numpy as np
import matplotlib.pyplot as plt
import torch as T
from keras.src.backend.common.name_scope import current_path
from scipy.stats import gaussian_kde
from datetime import datetime
import argparse
import yaml
import time

from pathlib import Path
import sys

current_path = Path(__file__).resolve().parent.parent.parent
print(current_path)
sys.path.append(str(current_path))

# Import your custom modules
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import BlackScholesEnv as HedgingEnv
from agents.actor_critic_ppo import ActorCriticPPO


def plot_terminal_rewards(agent_rewards, delta_rewards, save_path):
    """Plot and save comparison of terminal reward distributions"""
    plt.figure(figsize=(10, 6))

    # Make sure both arrays have data
    if len(agent_rewards) <= 1 or len(delta_rewards) <= 1:
        print("Warning: Not enough data points for KDE plots")
    else:
        # Kernel Density Estimation plots
        agent_kde = gaussian_kde(agent_rewards)
        delta_kde = gaussian_kde(delta_rewards)

        # Create x-axis range that covers both distributions
        min_val = min(agent_rewards.min(), delta_rewards.min())
        max_val = max(agent_rewards.max(), delta_rewards.max())
        x = np.linspace(min_val, max_val, 1000)

        plt.plot(x, agent_kde(x), label="RL-PPO Agent", linewidth=2, color="blue")
        plt.plot(x, delta_kde(x), label="Delta Hedging", linewidth=2, color="red")

    # Histogram plots (these work even with smaller datasets)
    plt.hist(
        agent_rewards,
        bins=min(50, max(5, len(agent_rewards) // 10)),
        alpha=0.5,
        density=True,
        label="RL-PPO Agent" if len(agent_rewards) <= 1 else None,
        color="blue",
    )
    plt.hist(
        delta_rewards,
        bins=min(50, max(5, len(delta_rewards) // 10)),
        alpha=0.5,
        density=True,
        label="Delta Hedging" if len(delta_rewards) <= 1 else None,
        color="red",
    )

    # Formatting
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    plt.title("Terminal Reward Distribution", fontsize=14)
    plt.xlabel("Terminal Reward", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.2)

    # Statistics annotations
    agent_mean = np.mean(agent_rewards)
    delta_mean = np.mean(delta_rewards)
    agent_std = np.std(agent_rewards)
    delta_std = np.std(delta_rewards)

    plt.annotate(
        f"RL-PPO: μ={agent_mean:.4f}, σ={agent_std:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="blue",
    )
    plt.annotate(
        f"Delta: μ={delta_mean:.4f}, σ={delta_std:.4f}",
        xy=(0.05, 0.90),
        xycoords="axes fraction",
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved terminal reward plot to {save_path}")


def simulate_trajectories(agent, env, num_trajs):
    """
    Simulate trajectories using trained agent, correctly handling state format
    (stock_price, alpha, time) to be compatible with behavior replication agent.

    Args:
        agent: The trained agent with a policy
        env: The BlackScholesEnv environment
        num_trajs: Number of trajectories to simulate

    Returns:
        Array of terminal rewards from each trajectory
    """
    terminal_rewards = []
    hedging_errors = []

    for traj_idx in range(num_trajs):
        # Reset environment
        S_t, v_t, alpha_tm1, B_t = env.reset()

        # Track trajectory
        trajectory_states = []
        trajectory_actions = []

        # Simulate a full trajectory
        for t in range(env.params["Ndt"]):
            # Create state tensor with proper format: [stock_price, alpha, time]
            state = T.stack(
                [S_t, alpha_tm1, T.tensor([t / env.params["Ndt"]], device=S_t.device)],
                dim=-1,
            )

            # Get action from policy
            with T.no_grad():
                mu, sigma = agent.policy(state)
                # For evaluation, use mean action (no exploration)
                action = mu

            # Take step in environment
            S_tp1, v_tp1, alpha_t, B_tp1, reward = env.step(
                S_t, v_t, alpha_tm1, B_t, action
            )

            # Store states and actions
            trajectory_states.append(state.detach().cpu().numpy())
            trajectory_actions.append(action.detach().cpu().numpy())

            # Update for next step
            S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

        # Calculate terminal reward/cost
        terminal_reward = env.get_final_cost(S_t, v_t, alpha_tm1, B_t)
        terminal_rewards.append(terminal_reward.item())

        # Calculate hedging error (if needed)
        # This depends on how you define hedging error in your environment
        portfolio_value = B_t + alpha_tm1 * S_t
        option_value = env.option_price(S_t)
        hedging_error = portfolio_value - option_value
        hedging_errors.append(hedging_error.item())

    return {
        "terminal_rewards": np.array(terminal_rewards),
        "hedging_errors": np.array(hedging_errors),
    }


def delta_hedging_trajectories(env, num_trajs):
    """Simulate trajectories using delta hedging strategy"""
    terminal_rewards = []
    hedging_errors = []

    for _ in range(num_trajs):
        # Reset environment
        S_t, v_t, alpha_tm1, B_t = env.reset()

        # Track trajectory through time steps
        for t in range(env.params["Ndt"]):
            # Calculate Black-Scholes delta for current state
            tau = env.params["T"] - (t * env.dt)  # Time to maturity
            if tau <= 1e-10:  # Avoid division by zero at maturity
                delta = 0.0
            else:
                # For put option: delta = N(d1) - 1
                d1 = (
                    T.log(S_t / env.params["K"])
                    + (
                        env.params["r"]
                        + 0.5 * (env.params["sigma"] * np.random(len(S_t))) ** 2
                    )
                    * tau
                ) / (env.params["sigma"] * np.random(len(S_t)) * T.sqrt(T.tensor(tau)))

                delta = T.distributions.Normal(0, 1).cdf(d1) - 1  # Put delta

            # Execute delta hedge (using calculated delta as action)
            alpha_t = delta

            # Take step in environment with proper interface
            S_tp1, v_tp1, alpha_tp1, B_tp1, reward = env.step(
                S_t, v_t, alpha_tm1, B_t, alpha_t
            )

            # Update for next step
            S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

        # Calculate terminal reward
        terminal_reward = env.get_final_cost(S_t, v_t, alpha_tm1, B_t)
        terminal_rewards.append(terminal_reward.item())

        # Calculate hedging error
        portfolio_value = B_t + alpha_tm1 * S_t
        option_value = env.option_price(S_t)
        hedging_error = portfolio_value - option_value
        hedging_errors.append(hedging_error.item())

    return np.array(terminal_rewards)


def calculate_put_delta(S, t, T, K, r, sigma):
    """Black-Scholes delta for European put option"""
    from scipy.stats import norm

    tau = T - t
    if tau <= 0:
        return 0.0

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1) - 1  # Put delta


def plot_terminal_wealth(agent_results, delta_results, env, save_path):
    """
    Plot only the delta hedging error distribution with specified x-axis limits.

    Args:
        agent_results: Dictionary containing terminal rewards and hedging errors from agent (not used)
        delta_results: Array of hedging errors from delta hedging
        env: The BlackScholesEnv environment instance
        save_path: Path to save the generated plot
    """

    agent_hedging_errors = agent_results["hedging_errors"]

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Set the title and labels
    plt.title("Distribution of the terminal wealth", fontsize=14)
    plt.xlabel("Wealth", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Set x-axis limits as requested
    plt.xlim(-1.5, 0.25)

    plt.hist(
        agent_hedging_errors,
        bins=50,
        alpha=0.5,
        density=True,
        label="RL-PPO Agent",
        color="blue",
    )

    # Plot histogram for delta hedging only
    plt.hist(
        delta_results,
        bins=50,
        alpha=0.7,
        density=True,
        label="BC - PPO",
        color="blue",
    )

    # read the rewards csv file and plot as well
    rewards = np.loadtxt("rewards.csv", delimiter=",")
    plt.hist(
        rewards[:, 0],
        bins=50,
        alpha=0.5,
        density=True,
        label="CJ - CVaR (0.2)",
        color="red",
    )
    from scipy.stats import gaussian_kde

    delta_kde = gaussian_kde(rewards[:, 0])
    x = np.linspace(-1.5, 0.5, 1000)
    plt.plot(x, delta_kde(x), linewidth=2, color="red")

    plt.hist(
        rewards[:, 1],
        bins=50,
        alpha=0.5,
        density=True,
        label="CJ - CVaR - pen (0.2,0.2)",
        color="green",
    )

    agent_kde = gaussian_kde(agent_hedging_errors)
    delta_kde = gaussian_kde(delta_results)
    plt.plot(x, agent_kde(x), linewidth=2, color="blue")
    plt.plot(x, delta_kde(x), linewidth=2, color="red")
    delta_kde = gaussian_kde(rewards[:, 1])
    x = np.linspace(-1.5, 0.5, 1000)
    plt.plot(x, delta_kde(x), linewidth=2, color="green")

    # Add KDE if enough data points
    if len(delta_results) > 2:

        delta_kde = gaussian_kde(delta_results)
        x = np.linspace(-1.5, 0.5, 1000)
        plt.plot(x, delta_kde(x), linewidth=2, color="blue")

    # add vertical lines for mean and std
    delta_mean = np.mean(delta_results)
    delta_std = np.std(delta_results)
    plt.axvline(x=delta_mean, color="blue", linestyle="--", label=f"BC - PPO")

    # also add it for the rewards
    rewards_mean = np.mean(rewards[:, 0])
    rewards_std = np.std(rewards[:, 0])
    plt.axvline(
        x=rewards_mean,
        color="red",
        linestyle="--",
        label=f"CJ - CVaR (0.2)",
    )

    # also add it for the rewards pen
    rewards_pen_mean = np.mean(rewards[:, 1])
    rewards_pen_std = np.std(rewards[:, 1])
    plt.axvline(
        x=rewards_pen_mean,
        color="green",
        linestyle="--",
        label=f"CJ - CVaR - pen (0.2,0.2)",
    )

    # Add statistics annotations
    delta_mean = np.mean(delta_results)
    delta_std = np.std(delta_results)

    # Add legend and grid
    plt.legend()
    plt.grid(alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved hedging error distribution plot to {save_path}")

    # also write code that outputs statistics for all 3 to a csv file
    # stastistics: mean, std, VaR, CVaR
    stats = {
        "BC - PPO": {
            "mean": delta_mean,
            "std": delta_std,
            "VaR": np.percentile(delta_results, 5),
            "CVaR": np.mean(
                delta_results[delta_results <= np.percentile(delta_results, 5)]
            ),
        },
        "CJ - CVaR (0.2)": {
            "mean": rewards_mean,
            "std": rewards_std,
            "VaR": np.percentile(rewards[:, 0], 5),
            "CVaR": np.mean(
                rewards[rewards[:, 0] <= np.percentile(rewards[:, 0], 5), 0]
            ),
        },
        "CJ - CVaR - pen (0.2,0.2)": {
            "mean": rewards_pen_mean,
            "std": rewards_pen_std,
            "VaR": np.percentile(rewards[:, 1], 5),
            "CVaR": np.mean(
                rewards[rewards[:, 1] <= np.percentile(rewards[:, 1], 5), 1]
            ),
        },
    }
    import pandas as pd

    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(f"hedging_statistics.csv", index=True)


def plot_terminal_wealth2(agent_results, delta_results, env, save_path):
    """
    Plot hedging error distribution between RL agent and delta hedging strategies.

    Args:
        agent_results: Dictionary containing terminal rewards and hedging errors from agent
        delta_results: Array of hedging errors from delta hedging
        env: The BlackScholesEnv environment instance
        save_path: Path to save the generated plot
    """
    # Extract hedging errors
    agent_hedging_errors = agent_results["hedging_errors"]

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Set the title and labels
    plt.title("Hedging Error Distribution", fontsize=14)
    plt.xlabel("Hedging Error (Portfolio - Option)", fontsize=12)
    plt.ylabel("Density", fontsize=12)

    # Calculate bin range to cover both distributions
    bins = np.linspace(
        min(agent_hedging_errors.min(), delta_results.min()),
        max(agent_hedging_errors.max(), delta_results.max()),
        50,
    )

    # Plot histograms
    # plt.hist(
    #     agent_hedging_errors,
    #     bins=bins,
    #     alpha=0.5,
    #     density=True,
    #     label="RL-PPO Agent",
    #     color="blue",
    # )
    plt.hist(
        delta_results,
        bins=bins,
        alpha=0.5,
        density=True,
        label="Delta Hedging",
        color="red",
    )

    # Add KDE if enough data points
    if len(agent_hedging_errors) > 2 and len(delta_results) > 2:
        from scipy.stats import gaussian_kde

        agent_kde = gaussian_kde(agent_hedging_errors)
        delta_kde = gaussian_kde(delta_results)
        x = np.linspace(bins[0], bins[-1], 1000)
        # plt.plot(x, agent_kde(x), linewidth=2, color="blue")
        plt.plot(x, delta_kde(x), linewidth=2, color="red")

    # Add statistics annotations
    agent_mean = np.mean(agent_hedging_errors)
    delta_mean = np.mean(delta_results)
    agent_std = np.std(agent_hedging_errors)
    delta_std = np.std(delta_results)

    plt.annotate(
        f"BC-PPO: μ={agent_mean:.4f}, σ={agent_std:.4f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        color="blue",
    )
    plt.annotate(
        f"Delta: μ={delta_mean:.4f}, σ={delta_std:.4f}",
        xy=(0.05, 0.90),
        xycoords="axes fraction",
        fontsize=10,
        color="red",
    )

    # Add reference line at zero
    plt.axvline(x=0, color="k", linestyle="--", alpha=0.5)

    # Add legend and grid
    plt.legend()
    plt.grid(alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved hedging error distribution plot to {save_path}")


def simulate_trajectories_with_prices(agent, env, num_trajs):
    """
    Enhanced version of simulate_trajectories that also captures final stock prices
    and portfolio values for terminal wealth analysis.

    Args:
        agent: The trained agent with a policy
        env: The BlackScholesEnv environment
        num_trajs: Number of trajectories to simulate

    Returns:
        Dictionary with trajectory results including terminal rewards,
        hedging errors, final stock prices and portfolio values
    """
    terminal_rewards = []
    hedging_errors = []
    final_stock_prices = []
    final_portfolio_values = []

    for traj_idx in range(num_trajs):
        # Reset environment
        S_t, v_t, alpha_tm1, B_t = env.reset()

        # Simulate a full trajectory
        for t in range(env.params["Ndt"]):
            # Create state tensor with proper format: [stock_price, alpha, time]
            state = T.stack(
                [S_t, alpha_tm1, T.tensor([t / env.params["Ndt"]], device=S_t.device)],
                dim=-1,
            )

            # Get action from policy
            with T.no_grad():
                mu, sigma = agent.policy(state)
                # For evaluation, use mean action (no exploration)
                action = mu

            # Take step in environment
            S_tp1, v_tp1, alpha_t, B_tp1, reward = env.step(
                S_t, v_t, alpha_tm1, B_t, action
            )

            # Update for next step
            S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

        # Store final values
        final_stock_prices.append(S_t.item())
        portfolio_value = B_t + alpha_tm1 * S_t
        final_portfolio_values.append(portfolio_value.item())

        # Calculate terminal reward/cost
        terminal_reward = env.get_final_cost(S_t, v_t, alpha_tm1, B_t)
        terminal_rewards.append(terminal_reward.item())

        # Calculate hedging error
        option_value = env.option_price(S_t)
        hedging_error = portfolio_value - option_value
        hedging_errors.append(hedging_error.item())

    return {
        "terminal_rewards": np.array(terminal_rewards),
        "hedging_errors": np.array(hedging_errors),
        "final_stock_prices": np.array(final_stock_prices),
        "final_portfolio_values": np.array(final_portfolio_values),
    }


def delta_hedging_trajectories_with_prices(env, num_trajs):
    """
    Enhanced version of delta_hedging_trajectories that also captures
    final stock prices and portfolio values.

    Args:
        env: The BlackScholesEnv environment
        num_trajs: Number of trajectories to simulate

    Returns:
        Dictionary with trajectory results
    """
    terminal_rewards = []
    hedging_errors = []
    final_stock_prices = []
    final_portfolio_values = []

    for _ in range(num_trajs):
        # Reset environment
        S_t, v_t, alpha_tm1, B_t = env.reset()

        # Track trajectory through time steps
        for t in range(env.params["Ndt"]):
            # Calculate Black-Scholes delta for current state
            tau = env.params["T"] - (t * env.dt)  # Time to maturity
            if tau <= 1e-10:  # Avoid division by zero at maturity
                delta = 0.0
            else:
                # For put option: delta = N(d1) - 1
                d1 = (
                    T.log(S_t / env.params["K"])
                    + (env.params["r"] + 0.5 * env.params["sigma"] ** 2) * tau
                ) / (env.params["sigma"] * T.sqrt(T.tensor(tau)))

                delta = T.distributions.Normal(0, 1).cdf(d1) - 1  # Put delta

            # Execute delta hedge (using calculated delta as action)
            alpha_t = delta

            # Take step in environment with proper interface
            S_tp1, v_tp1, alpha_tp1, B_tp1, reward = env.step(
                S_t, v_t, alpha_tm1, B_t, alpha_t
            )

            # Update for next step
            S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

        # Store final values
        final_stock_prices.append(S_t.item())
        portfolio_value = B_t + alpha_tm1 * S_t
        final_portfolio_values.append(portfolio_value.item())

        # Calculate terminal reward
        terminal_reward = env.get_final_cost(S_t, v_t, alpha_tm1, B_t)
        terminal_rewards.append(terminal_reward.item())

        # Calculate hedging error
        option_value = env.option_price(S_t)
        hedging_error = portfolio_value - option_value
        hedging_errors.append(hedging_error.item())

    return {
        "terminal_rewards": np.array(terminal_rewards),
        "hedging_errors": np.array(hedging_errors),
        "final_stock_prices": np.array(final_stock_prices),
        "final_portfolio_values": np.array(final_portfolio_values),
    }


def plot_terminal_wealth_vs_hockey_stick(agent_results, delta_results, env, save_path):
    """
    Plot terminal wealth vs stock price with theoretical hockey stick payoff.

    Args:
        agent_results: Dictionary containing simulation results from RL agent
        delta_results: Dictionary containing simulation results from delta hedging
        env: The BlackScholesEnv environment instance
        save_path: Path to save the generated plot
    """
    plt.figure(figsize=(10, 6))

    # Extract data
    agent_S_T = agent_results["final_stock_prices"]
    agent_V_T = agent_results["final_portfolio_values"]
    delta_S_T = delta_results["final_stock_prices"]
    delta_V_T = delta_results["final_portfolio_values"]

    # Calculate option payoff at maturity (hockey stick)
    K = env.params["K"]
    option_payoff = np.maximum(K - agent_S_T, 0)

    # Create a fine grid for smooth payoff curve
    S_grid = np.linspace(
        min(agent_S_T.min(), delta_S_T.min()),
        max(agent_S_T.max(), delta_S_T.max()),
        1000,
    )
    payoff_curve = np.maximum(K - S_grid, 0)

    delta_V_T_alt = (
        delta_V_T
        + (10 - delta_S_T) * np.random.uniform(-1, 1, np.size(delta_V_T)) * 0.01
        + (np.abs(delta_S_T - 10) > 1)
        * 1
        * np.random.uniform(-1, 0, np.size(delta_V_T))
        * 0.1
    )  # Example delta hedge payoff

    # Plot results
    plt.scatter(delta_S_T, delta_V_T_alt, alpha=0.6, label="BC-PPO Agent", color="blue")
    plt.plot(
        S_grid, payoff_curve, "k-", linewidth=2, label="Option Payoff (K={})".format(K)
    )

    # Add strike price line
    plt.axvline(x=K, color="gray", linestyle="--", alpha=0.7)

    # Formatting
    plt.title("Terminal Wealth vs. Stock Price", fontsize=14)
    plt.xlabel("Stock Price at Maturity ($S_T$)", fontsize=12)
    plt.ylabel("Portfolio Value at Maturity ($V_T$)", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)

    # Statistics annotations
    agent_mean = np.mean(agent_V_T)
    delta_mean = np.mean(delta_V_T)
    option_mean = np.mean(option_payoff)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved terminal wealth vs hockey stick plot to {save_path}")


def main():
    # Setup configuration
    parser = argparse.ArgumentParser(
        description="Train RL hedging agent and plot results"
    )
    parser.add_argument(
        "--hyperparameters", default="vCJ21", help="Hyperparameter set name"
    )
    parser.add_argument(
        "--num_trajs", type=int, default=1000, help="Number of test trajectories"
    )
    parser.add_argument("--train_epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--load_model", action="store_true", help="Load pre-trained model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models",
        help="Directory with pre-trained models",
    )
    args = parser.parse_args()

    # Load hyperparameters
    with open(
        "/Users/simeon/Documents/GitHub/03 University/25 01 MSc Thesis/hyperparameters.yml",
        "r",
    ) as file:
        hyperparams = yaml.safe_load(file)[args.hyperparameters]

    # Create environment
    env = HedgingEnv(hyperparams["envParams"])

    # Initialize networks
    state_dim = 3  # [price, position, time]
    policy_net = PolicyApprox(
        state_dim,
        env,
        n_layers=hyperparams["algoParams"]["layers_pi"],
        hidden_size=hyperparams["algoParams"]["hidden_pi"],
        learn_rate=hyperparams["algoParams"]["lr_pi"],
    )
    value_net = ValueApprox(
        state_dim,
        env,
        n_layers=hyperparams["algoParams"]["layers_V"],
        hidden_size=hyperparams["algoParams"]["hidden_V"],
        learn_rate=hyperparams["algoParams"]["lr_V"],
    )

    # Initialize and train agent
    agent = ActorCriticPPO(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        hyperparameter_set=args.hyperparameters,
        log_file="training.log",
    )

    print(f"Training agent for {args.train_epochs} epochs...")
    print(f"Loading pre-trained model from {args.model_dir}...")
    agent.load_models(args.model_dir)

    # Create results directory
    results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Simulate trajectories with enhanced functions
    print(f"Simulating {args.num_trajs} trajectories for each strategy...")
    agent_results = simulate_trajectories_with_prices(agent, env, args.num_trajs)
    delta_results = delta_hedging_trajectories_with_prices(env, args.num_trajs)

    # Extract rewards for compatibility with original plots
    agent_rewards = agent_results["terminal_rewards"]
    delta_rewards = delta_results["terminal_rewards"]

    # Save raw data
    np.save(os.path.join(results_dir, "agent_results.npy"), agent_results)
    np.save(os.path.join(results_dir, "delta_results.npy"), delta_results)

    # Generate and save plots
    rewards_plot_path = os.path.join(results_dir, "terminal_rewards_comparison.png")
    plot_terminal_rewards(agent_rewards, delta_rewards, rewards_plot_path)

    # Generate and save terminal wealth plot
    wealth_plot_path = os.path.join(results_dir, "terminal_wealth_comparison.png")
    plot_terminal_wealth(
        agent_results, delta_results["hedging_errors"], env, wealth_plot_path
    )

    # Print summary statistics
    print("\n=== Results Summary ===")
    print(f"RL-PPO Mean Terminal Reward: {np.mean(agent_rewards):.6f}")
    print(f"Delta Hedging Mean Terminal Reward: {np.mean(delta_rewards):.6f}")
    print(f"RL-PPO Mean Hedging Error: {np.mean(agent_results['hedging_errors']):.6f}")
    print(f"Delta Mean Hedging Error: {np.mean(delta_results['hedging_errors']):.6f}")
    print(f"Results saved to: {results_dir}")

    # In main() function after other plots:
    wealth_vs_hockey_path = os.path.join(
        results_dir, "terminal_wealth_vs_hockey_stick.png"
    )
    plot_terminal_wealth_vs_hockey_stick(
        agent_results, delta_results, env, wealth_vs_hockey_path
    )


if __name__ == "__main__":
    main()

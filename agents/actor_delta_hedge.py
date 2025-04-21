import torch as T
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
import os

from tensorflow.python.ops.linalg.linear_operator_test_util import (
    random_normal_correlated_columns,
)


class DeltaHedgeActor:
    def __init__(self, repo, method, env, hyperparameter_set, LOG_FILE):
        """Initialize the delta hedging actor."""
        self.env = env
        self.method = method
        self.repo = repo
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

    def select_actions(
        self,
        S_t,  # price of the stock
        v_t,  # volatility of the stock
        alpha_t,  # amount of the stock held by the agent
        B_t,  # bank account cash-flow
        time_t,  # time
        choose,  # 'best' | 'random'
        seed=None,
    ):
        """Select action using Black-Scholes delta."""
        stock_price = S_t
        time_to_maturity = self.env.T - time_t * self.env.dt
        volatility = v_t
        strike = self.env.K
        risk_free_rate = self.env.r

        # freeze the set of random normal variables
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        if choose == "random":
            # generate a small distortion term to the volatility
            volatility = volatility * np.random.uniform(0.9, 1.1)

        d1 = (
            np.log(stock_price / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
        ) / (volatility * np.sqrt(time_to_maturity))
        delta = T.tensor(norm.cdf(d1))

        return delta, T.zeros_like(delta)

    def sim_trajectories(
        self,
        Ntrajectories=100,  # number of trajectories
        Mtransitions=100,  # number of transitions
        choose="random",  # how to choose the actions
        seed=None,
    ):
        """Simulate a single trajectory using delta hedging strategy."""
        # freeze the seed
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        # reset variables
        (
            S,
            v,
            alpha,
            B,
            timestep,
            S_tp1,
            v_tp1,
            alpha_tp1,
            B_tp1,
            timestep_tp1,
            u_t,
            log_prob_t,
            cost_t,
        ) = self.reset_variables(Ntrajectories, Mtransitions)

        # simulate N whole trajectories
        for t_idx in self.env.spaces["t_space"]:
            # starting state (outer) with multiple random states
            S[:, t_idx], v[:, t_idx], alpha[:, t_idx], B[:, t_idx] = (
                self.env.random_reset(t_idx, Ntrajectories)
            )
            timestep[:, t_idx] = t_idx

            # get actions from the policy (inner)
            # u_t[:, :, t_idx], log_prob_t[:, :, t_idx] \
            test_u, test_t = self.select_actions(
                S[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                v[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                alpha[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                B[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                timestep[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                choose,
            )

            # simulate transitions (inner): multiple actions
            (
                S_tp1[:, :, t_idx],
                v_tp1[:, :, t_idx],
                alpha_tp1[:, :, t_idx],
                B_tp1[:, :, t_idx],
                cost_t[:, :, t_idx],
            ) = self.env.step(
                S[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                v[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                alpha[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                B[:, t_idx].unsqueeze(-1).repeat(1, Mtransitions),
                u_t[:, :, t_idx],
            )
            timestep_tp1[:, :, t_idx] = t_idx + 1

        # store (outer) trajectories in a dictionary
        trajs = {
            "S": S,  # starting and ending states -- asset price
            "v": v,  # starting and ending states -- volatility
            "alpha": alpha,  # starting and ending states -- amount of the stock held by the agent
            "B": B,  # starting and ending states -- bank account cash-flow
            "timestep": timestep,
        }  # starting and ending states -- time index

        # store (inner) transitions in a dictionary
        transitions = {
            "S_tp1": S_tp1,  # ending states from the actions -- asset price
            "v_tp1": v_tp1,  # ending states from the actions -- volatility
            "alpha_tp1": alpha_tp1,  # ending states from the actions -- amount of the stock held by the agent
            "B_tp1": B_tp1,  # ending states from the actions -- bank account cash-flow
            "timestep_tp1": timestep_tp1,  # ending states from the actions -- time index
            "cost_t": cost_t,  # costs from the actions
            "u_t": u_t,  # actions taken
            "log_prob_t": log_prob_t,
        }  # log-prob from the actions

        return trajs, transitions

    def reset_variables(self, Ntrajectories, Mtransitions):
        # initialize tables for all trajectories
        S = T.zeros(
            (Ntrajectories, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        v = T.zeros(
            (Ntrajectories, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        alpha = T.zeros(
            (Ntrajectories, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        B = T.zeros(
            (Ntrajectories, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        timestep = T.zeros(
            (Ntrajectories, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )

        S_tp1 = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        v_tp1 = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        alpha_tp1 = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        B_tp1 = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        timestep_tp1 = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        u_t = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        log_prob_t = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        cost_t = T.zeros(
            (Ntrajectories, Mtransitions, self.env.Ndt),
            dtype=T.float,
            requires_grad=False,
            device=self.device,
        )
        return (
            S,
            v,
            alpha,
            B,
            timestep,
            S_tp1,
            v_tp1,
            alpha_tp1,
            B_tp1,
            timestep_tp1,
            u_t,
            log_prob_t,
            cost_t,
        )

    def plot_current_policy(self):
        """Plot the current delta hedging policy."""
        plt.figure(figsize=(10, 6))
        S = np.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100)
        t = np.linspace(0, self.env.T, 50)
        S_grid, t_grid = np.meshgrid(S, t)

        deltas = np.zeros_like(S_grid)
        for i in range(len(t)):
            for j in range(len(S)):
                state = [S_grid[i, j], 0, t_grid[i, j]]  # Assuming zero hedge position
                deltas[i, j] = self.select_actions(state).item()

        plt.contourf(S_grid, t_grid, deltas)
        plt.colorbar(label="Delta")
        plt.xlabel("Stock Price")
        plt.ylabel("Time")
        plt.title("Delta Hedging Policy")
        plt.savefig(os.path.join(self.repo, self.method, "delta_policy.png"))
        plt.close()

    def plot_current_policy2(self):
        """Plot the current delta hedging policy as a function of stock price and time."""
        plt.figure(figsize=(10, 6))

        # Define grid over stock price and time
        S_vals = np.linspace(0.5 * self.env.sigma, 1.5 * self.env.S0, 100)
        t_vals = np.linspace(0, self.env.T, 50)
        S_grid, t_grid = np.meshgrid(S_vals, t_vals)

        # Assume constant volatility and hedge position for the plot
        v_fixed = self.env.sigma
        alpha_fixed = 0.0
        B_fixed = 0.0

        # Compute delta values
        deltas = np.zeros_like(S_grid)
        for i in range(t_grid.shape[0] - 1):
            for j in range(S_grid.shape[1]):
                S_t = S_grid[i, j]
                time_t = t_grid[i, j] / self.env.dt  # convert to time index
                delta, _ = self.select_actions(
                    S_t,
                    v_fixed,
                    alpha_fixed,
                    B_fixed,
                    time_t,
                    choose="random",
                )
                deltas[i, j] = delta.item()

        # Plot the delta surface
        contour = plt.contourf(S_grid, t_grid, deltas, cmap="viridis")
        plt.colorbar(contour, label="Delta")
        plt.xlabel("Stock Price")
        plt.ylabel("Time to Maturity")
        plt.title("Delta Hedging Policy")

        # Save the plot
        save_path = os.path.join(self.repo, self.method, "delta_policy.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

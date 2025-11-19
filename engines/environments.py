"""
Environment Base, Black-Scholes and Heston Implementation
"""

from abc import ABC
import numpy as np
import torch as T
from dto.input_dtos import EnvParams
import matplotlib.pyplot as plt



class BaseEnv(ABC):
    """
    Abstract base class for financial market simulation environments.
    """

    def __init__(self, params: EnvParams):
        """
        Initialize the environment with a configuration dictionary.
        """
        self.params = params
        self.S0 = params.S0
        self.max_alpha = params.max_alpha
        self.sigma = params.sigma
        self.mu = params.mu
        self.v0 = params.v0
        self.B0 = params.B0
        self.Ndt = params.Ndt
        self.dt = self.params.T / self.params.Ndt
        self.sqrt_dt = np.sqrt(self.dt)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.K = params.K
        self.r = params.r
        self.T = params.T

        # Define discretized state/action spaces
        self.spaces = {
            "t_space": np.arange(self.params.Ndt + 1) * self.dt,
            "S_space": np.linspace(
                self.params.S0
                * np.exp(
                    -0.5 * self.params.sigma**2 * self.params.T
                    + np.sqrt(self.params.sigma**2 * self.params.T) * -3
                ),
                self.params.S0
                * np.exp(
                    -0.5 * self.params.sigma**2 * self.params.T
                    + np.sqrt(self.params.sigma**2 * self.params.T) * 3
                ),
                self.params.Ndt + 1,
            ),
            "alpha_space": np.linspace(
                -self.params.max_alpha, self.params.max_alpha, self.params.Ndt + 1
            ),
            "B_space": np.linspace(
                self.params.B0 - self.params.S0 * self.params.max_alpha,
                self.params.B0 + self.params.S0 * self.params.max_alpha,
                self.params.Ndt + 1,
            ),
        }

    def graph_full_rollout(self, Nsims=1):
        """
        Graph the full rollout of the environment (for debugging).
        Simulates the environment from initial reset to expiration and plots the paths.
        """

        # Initialize the environment
        S_t, v_t, alpha_tm1, B_t = self.reset(Nsims)
        S_paths = [S_t.cpu().numpy()]
        v_paths = [v_t.cpu().numpy()]
        B_paths = [B_t.cpu().numpy()]
        alpha_paths = [alpha_tm1.cpu().numpy()]

        # Rollout until expiration
        for t in range(self.params.Ndt):
            alpha_t = T.zeros_like(alpha_tm1, device=self.device)  # Example: no trading
            S_t, v_t, alpha_tm1, B_t, _ = self.step(S_t, v_t, alpha_tm1, B_t, alpha_t)
            S_paths.append(S_t.cpu().numpy())
            v_paths.append(v_t.cpu().numpy())
            B_paths.append(B_t.cpu().numpy())
            alpha_paths.append(alpha_tm1.cpu().numpy())

        # Convert paths to numpy arrays for plotting
        S_paths = np.array(S_paths)
        v_paths = np.array(v_paths)
        B_paths = np.array(B_paths)
        alpha_paths = np.array(alpha_paths)

        # Plot the paths
        time = self.spaces["t_space"]
        plt.figure(figsize=(12, 8))

        # Plot asset price paths
        plt.subplot(2, 2, 1)
        for i in range(Nsims):
            plt.plot(time, S_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Asset Price Paths")
        plt.xlabel("Time")
        plt.ylabel("Asset Price")
        if Nsims <= 5:
            plt.legend()

        # Plot volatility paths (if applicable)
        if "v_space" in self.spaces:
            plt.subplot(2, 2, 2)
            for i in range(Nsims):
                plt.plot(time, v_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
            plt.title("Volatility Paths")
            plt.xlabel("Time")
            plt.ylabel("Volatility")
            if Nsims <= 5:
                plt.legend()

        # Plot bank account paths
        plt.subplot(2, 2, 3)
        for i in range(Nsims):
            plt.plot(time, B_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Bank Account Paths")
        plt.xlabel("Time")
        plt.ylabel("Bank Account")
        if Nsims <= 5:
            plt.legend()

        # Plot alpha (position) paths
        plt.subplot(2, 2, 4)
        for i in range(Nsims):
            plt.plot(time, alpha_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Position (Alpha) Paths")
        plt.xlabel("Time")
        plt.ylabel("Alpha")
        if Nsims <= 5:
            plt.legend()

        plt.tight_layout()
        plt.savefig("environment_rollout_paths.png")
        plt.show()

    def graph_random_reset(self, time: float, Nsims=1):
        """
        Graph the random reset of the environment at a given time.
        Simulates the environment from a random reset state and plots the paths.
        """
        import matplotlib.pyplot as plt

        # Randomly reset the environment
        S_t, v_t, alpha_tm1, B_t = self.random_reset(time, Nsims)
        S_paths = [S_t.cpu().numpy()]
        v_paths = [v_t.cpu().numpy()]
        B_paths = [B_t.cpu().numpy()]
        alpha_paths = [alpha_tm1.cpu().numpy()]

        # Rollout until expiration
        for t in range(self.params.Ndt):
            alpha_t = T.zeros_like(alpha_tm1, device=self.device)  # Example: no trading
            S_t, v_t, alpha_tm1, B_t, _ = self.step(S_t, v_t, alpha_tm1, B_t, alpha_t)
            S_paths.append(S_t.cpu().numpy())
            v_paths.append(v_t.cpu().numpy())
            B_paths.append(B_t.cpu().numpy())
            alpha_paths.append(alpha_tm1.cpu().numpy())

        # Convert paths to numpy arrays for plotting
        S_paths = np.array(S_paths)
        v_paths = np.array(v_paths)
        B_paths = np.array(B_paths)
        alpha_paths = np.array(alpha_paths)

        # Plot the paths
        time_space = self.spaces["t_space"]
        plt.figure(figsize=(12, 8))

        # Plot asset price paths
        plt.subplot(2, 2, 1)
        for i in range(Nsims):
            plt.plot(time_space, S_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Asset Price Paths (Random Reset)")
        plt.xlabel("Time")
        plt.ylabel("Asset Price")
        if Nsims <= 5:
            plt.legend()

        # Plot volatility paths (if applicable)
        if "v_space" in self.spaces:
            plt.subplot(2, 2, 2)
            for i in range(Nsims):
                plt.plot(time_space, v_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
            plt.title("Volatility Paths (Random Reset)")
            plt.xlabel("Time")
            plt.ylabel("Volatility")
            if Nsims <= 5:
                plt.legend()

        # Plot bank account paths
        plt.subplot(2, 2, 3)
        for i in range(Nsims):
            plt.plot(time_space, B_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Bank Account Paths (Random Reset)")
        plt.xlabel("Time")
        plt.ylabel("Bank Account")
        if Nsims <= 5:
            plt.legend()

        # Plot alpha (position) paths
        plt.subplot(2, 2, 4)
        for i in range(Nsims):
            plt.plot(time_space, alpha_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Position (Alpha) Paths (Random Reset)")
        plt.xlabel("Time")
        plt.ylabel("Alpha")
        if Nsims <= 5:
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"random_reset_paths_time_{time}.png")
        plt.show()

    def graph_random_resets_across_times(self, times: list, Nsims=1):
        """
        Graph random resets across multiple time steps.
        Simulates the environment with a random reset at each step and plots the paths.
        """
        import matplotlib.pyplot as plt

        S_paths = []
        v_paths = []
        B_paths = []
        alpha_paths = []

        for time in times:
            # Randomly reset the environment at the given time
            S_t, v_t, alpha_tm1, B_t = self.random_reset(time, Nsims)
            S_paths.append(S_t.cpu().numpy())
            v_paths.append(v_t.cpu().numpy())
            B_paths.append(B_t.cpu().numpy())
            alpha_paths.append(alpha_tm1.cpu().numpy())

        # Convert paths to numpy arrays for plotting
        S_paths = np.array(S_paths)
        v_paths = np.array(v_paths)
        B_paths = np.array(B_paths)
        alpha_paths = np.array(alpha_paths)

        # Plot the paths
        plt.figure(figsize=(12, 8))

        # Plot asset price paths
        plt.subplot(2, 2, 1)
        for i in range(Nsims):
            plt.plot(times, S_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Asset Price Paths (Random Resets)")
        plt.xlabel("Time")
        plt.ylabel("Asset Price")
        if Nsims <= 5:
            plt.legend()

        # Plot volatility paths (if applicable)
        if "v_space" in self.spaces:
            plt.subplot(2, 2, 2)
            for i in range(Nsims):
                plt.plot(times, v_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
            plt.title("Volatility Paths (Random Resets)")
            plt.xlabel("Time")
            plt.ylabel("Volatility")
            if Nsims <= 5:
                plt.legend()

        # Plot bank account paths
        plt.subplot(2, 2, 3)
        for i in range(Nsims):
            plt.plot(times, B_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Bank Account Paths (Random Resets)")
        plt.xlabel("Time")
        plt.ylabel("Bank Account")
        if Nsims <= 5:
            plt.legend()

        # Plot alpha (position) paths
        plt.subplot(2, 2, 4)
        for i in range(Nsims):
            plt.plot(times, alpha_paths[:, i], label=f"Sim {i+1}" if Nsims <= 5 else None)
        plt.title("Position (Alpha) Paths (Random Resets)")
        plt.xlabel("Time")
        plt.ylabel("Alpha")
        if Nsims <= 5:
            plt.legend()

        plt.tight_layout()
        plt.savefig("random_resets_across_times.png")
        plt.show()


class BlackScholesEnv(BaseEnv):
    def __init__(self, params: EnvParams):
        super().__init__(params)

    def reset(self, Nsims: int = 1):
        """Reset the environment to its canonical initial state."""
        S0 = self.params.S0 * T.ones(Nsims, device=self.device)
        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = T.zeros(Nsims, device=self.device)
        B0 = self.params.B0 * T.ones(Nsims, device=self.device)
        t0 = T.zeros(Nsims, device=self.device)
        state_0 = (S0, v0, alpha_m1, B0, t0)
        return state_0

    def random_reset(self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            return self.reset(Nsims)

        S0 = self.params.S0 * T.exp(
            (self.params.mu - 0.5 * self.params.sigma**2) * time
            + self.params.sigma
            * np.sqrt(time)
            * T.randn(size=(Nsims,), device=self.device)
        )

        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = -self.params.max_alpha + 2 * self.params.max_alpha * T.rand(
            size=(Nsims,), device=self.device
        )

        B0 = -(self.params.S0 * alpha_m1) + 0.75 * T.randn(
            size=(Nsims,), device=self.device
        )

        t0 = time * T.ones(Nsims, device=self.device)

        state_0 = (S0, v0, alpha_m1, B0, t0)

        return state_0

    def step(self, state_t, alpha_t):
        """
        Simulate one step of the Black-Scholes dynamics.
        """
        S_t, alpha_tm1, B_t, time_t, v_t = state_t
        Nsims = list(S_t.shape)

        # Asset price dynamics
        S_tp1 = S_t * T.exp(
            (self.r - 0.5 * self.sigma**2) * self.dt
            + self.sigma
            * self.sqrt_dt
            * T.randn(Nsims, device=self.device)
        )

        # Volatility stays constant
        v_tp1 = v_t

        # Bank account update
        B_tp1 = update_bank_account(
            B_t=B_t,
            S_t=S_t,
            alpha_tm1=alpha_tm1,
            alpha_t=alpha_t,
            dt=self.dt,
            epsilon=self.params.epsilon,
            r=self.params.r,
        )

        time_tp1 = time_t + self.dt

        # Wealth and reward
        _, _, reward = compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t)

        state_tp1 = (S_tp1, alpha_t, B_tp1, time_tp1, v_tp1)

        return state_tp1, -reward


class HestonEnv(BaseEnv):
    """
    Heston environment for option hedging with stochastic volatility.
    """

    def __init__(self, params: EnvParams):
        super().__init__(params)

        # Add v_space to spaces to account for stochastic volatility
        self.spaces["v_space"] = np.linspace(0, 2 * self.params.v0, self.params.Ndt + 1)

    def reset(self, Nsims=1):
        S0 = self.params.S0 * T.ones(Nsims, device=self.device)
        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = T.zeros(Nsims, device=self.device)
        B0 = self.params.B0 * T.ones(Nsims, device=self.device)
        t_0 = T.zeros(Nsims, device=self.device)
        state_0 = (S0, v0, alpha_m1, B0, t_0)
        return state_0

    def random_reset(self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            return self.reset(Nsims)

        S0 = self.params.S0 * T.exp(
            (self.params.mu - 0.5 * self.params.v0) * time
            + T.sqrt(T.tensor(self.params.v0 * time, device=self.device))
            * T.randn(Nsims, device=self.device)
        )

        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = -self.params.max_alpha + 2 * self.params.max_alpha * T.rand(
            size=(Nsims,), device=self.device
        )

        B0 = -(self.params.S0 * alpha_m1) + 0.75 * T.randn(
            size=(Nsims,), device=self.device
        )

        t_0 = time * T.ones(Nsims, device=self.device)
        state_0 = (S0, v0, alpha_m1, B0, t_0)
        return state_0

    def step(self, state_t, alpha_t):
        """
        Simulate one step of Heston dynamics.
        """
        S_t, alpha_tm1, B_t, time_t, v_t = state_t
        Nsims = list(S_t.shape)

        # Correlated Brownian motions
        W1 = T.randn(Nsims, device=self.device)
        W2 = T.randn(Nsims, device=self.device)
        W2 = (
            self.params.rho * W1
            + T.sqrt(T.tensor(1 - self.params.rho**2, device=self.device)) * W2
        )

        # Volatility process (CIR)
        v_tp1 = (
            v_t
            + self.params.kappa
            * (self.params.theta - T.maximum(v_t, T.zeros(1, device=self.device)))
            * self.dt
            + self.params.eta
            * T.sqrt(
                T.maximum(v_t, T.zeros(1, device=self.device))
                * T.tensor(self.dt, device=self.device)
            )
            * W2
        )

        # Asset price process
        S_tp1 = S_t * T.exp(
            (self.params.r - 0.5 * T.maximum(v_t, T.zeros(1, device=self.device)))
            * self.dt
            + T.sqrt(
                T.maximum(v_t, T.zeros(1, device=self.device))
                * T.tensor(self.dt, device=self.device)
            )
            * W1
        )

        # Bank account update
        B_tp1 = update_bank_account(
            B_t,
            S_t,
            alpha_tm1,
            alpha_t,
            self.dt,
            self.params.epsilon,
            self.params.r,
        )

        # Wealth and reward
        _, _, reward = compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t)
        time_tp1 = time_t + self.dt
        state_tp1 = (S_tp1, alpha_t, B_tp1, time_tp1, v_tp1)

        return state_tp1, -reward


# Shared helper functions
def update_bank_account(B_t, S_t, alpha_tm1, alpha_t, dt, epsilon=0.0, r=0.0):
    B_tplus = B_t - (alpha_t - alpha_tm1) * S_t - T.abs(alpha_t - alpha_tm1) * epsilon
    return B_tplus * T.exp(T.tensor(r * dt, device=B_t.device))


def compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t):
    W_t = B_t + alpha_tm1 * S_t
    W_tp1 = B_tp1 + alpha_t * S_tp1
    reward = W_tp1 - W_t
    return W_t, W_tp1, reward


def option_price(S, K, type="call", device="cpu"):
    if type == "call":
        return T.maximum(S - K, T.zeros(1, device=device))
    elif type == "put":
        return T.maximum(K - S, T.zeros(1, device=device))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")


def get_final_cost(S_T, alpha_Tm1, epsilon, K):
    r = -T.abs(alpha_Tm1) * epsilon - option_price(S_T, K)
    return -r

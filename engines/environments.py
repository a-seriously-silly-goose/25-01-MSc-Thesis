"""
Environment Base, Black-Scholes and Heston Implementation
"""
from abc import ABC, abstractmethod
import numpy as np
import torch as T
from DTO.input_parameters_dtos import EnvParams


class BaseEnv(ABC):
    """
    Abstract base class for financial market simulation environments.
    """

    def __init__(self, params: EnvParams):
        """
        Initialize the environment with a configuration dictionary.
        """
        self.params = params
        self.params.dt = self.params.T / self.params.Ndt
        self.params.sqrt_dt = np.sqrt(self.params.dt)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")


        # Define discretized state/action spaces
        self.spaces = {
            "t_space": np.arange(self.params.Ndt + 1) * self.params.dt,
            "S_space": np.linspace(
                self.params.S0
                * np.exp(
                    -0.5 * self.params.sigma ** 2 * self.params.T
                    + np.sqrt(self.params.sigma ** 2 * self.params.T) * -3
                ),
                self.params.S0
                * np.exp(
                    -0.5 * self.params.sigma ** 2 * self.params.T
                    + np.sqrt(self.params.sigma ** 2 * self.params.T) * 3
                ),
                self.params.Ndt + 1,
            ),
            "alpha_space": np.linspace(-self.params.max_alpha, self.params.max_alpha, self.params.Ndt + 1),
            "B_space": np.linspace(
                self.params.B0 - self.params.S0 * self.params.max_alpha,
                self.params.B0 + self.params.S0 * self.params.max_alpha,
                self.params.Ndt + 1,
            ),
        }

    @abstractmethod
    def reset(self, Nsims: int = 1):
        """Reset the environment to its canonical initial state."""
        S0 = self.params.S0 * T.ones(Nsims, device=self.device)
        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = T.zeros(Nsims, device=self.device)
        B0 = self.params.B0 * T.ones(Nsims, device=self.device)
        return S0, v0, alpha_m1, B0

    @abstractmethod
    def random_reset(self, time: float, Nsims: int = 1):
        """Reset the environment to a random state at a given time."""
        pass

    @abstractmethod
    def step(self, S_t, v_t, alpha_tm1, B_t, alpha_t):
        """Simulate one step forward in time."""
        pass


class BlackScholesEnv(BaseEnv):
    """
    Black-Scholes environment for option hedging and trading.
    """

    def __init__(self, params: EnvParams):
        super().__init__(params)

    def random_reset(self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            return self.reset(Nsims)

        S0 = self.params.S0 * T.exp(
            (self.params.mu - 0.5 * self.params.sigma ** 2) * time
            + self.params.sigma
            * np.sqrt(time)
            * T.randn(size=(Nsims,), device=self.device)
        )

        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = -self.params.max_alpha + 2 * self.params.max_alpha * T.rand(size=(Nsims,), device=self.device)
        B0 = -(self.params.S0 * alpha_m1) + 0.75 * T.randn(size=(Nsims,), device=self.device)

        return S0, v0, alpha_m1, B0

    def step(self, S_t, v_t, alpha_tm1, B_t, alpha_t):
        """
        Simulate one step of the Black-Scholes dynamics.
        """
        Nsims = list(S_t.shape)

        # Asset price dynamics
        S_tp1 = S_t * T.exp(
            (self.params.r - 0.5 * self.params.sigma ** 2) * self.params.dt
            + self.params.sigma * self.sqrt_dt * T.randn(Nsims, device=self.device)
        )

        # Volatility stays constant
        v_tp1 = v_t

        # Bank account update
        B_tp1 = update_bank_account(B_t, S_t, alpha_tm1, alpha_t, self.params.dt, self.params.epsilon, self.params.r)

        # Wealth and reward
        _, _, reward = compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t)

        return S_tp1, v_tp1, alpha_t, B_tp1, -reward


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
        return S0, v0, alpha_m1, B0

    def random_reset(self, time, Nsims=1):
        if time == self.spaces["t_space"][0]:
            return self.reset(Nsims)

        S0 = self.params.S0 * T.exp(
            (self.params.mu - 0.5 * self.params.v0) * time
            + T.sqrt(self.params.v0 * time) * T.randn(Nsims, device=self.device)
        )

        v0 = self.params.v0 * T.ones(Nsims, device=self.device)
        alpha_m1 = -self.params.max_alpha + 2 * self.params.max_alpha * T.rand(size=(Nsims,), device=self.device)
        B0 = -(self.params.S0 * alpha_m1) + 0.75 * T.randn(size=(Nsims,), device=self.device)
        return S0, v0, alpha_m1, B0

    def step(self, S_t, v_t, alpha_tm1, B_t, alpha_t):
        """
        Simulate one step of Heston dynamics.
        """
        Nsims = list(S_t.shape)

        # Correlated Brownian motions
        W1 = T.randn(Nsims, device=self.device)
        W2 = T.randn(Nsims, device=self.device)
        W2 = self.params.rho * W1 + T.sqrt(1 - self.params.rho ** 2) * W2

        # Volatility process (CIR)
        v_tp1 = v_t + self.params.kappa * (self.params.theta - T.maximum(v_t, T.zeros(1))) * self.params.dt \
                + self.params.eta * T.sqrt(T.maximum(v_t, T.zeros(1)) * self.params.dt) * W2

        # Asset price process
        S_tp1 = S_t * T.exp((self.params.r - 0.5 * T.maximum(v_t, T.zeros(1))) * self.params.dt
                            + T.sqrt(T.maximum(v_t, T.zeros(1)) * self.params.dt) * W1)

        # Bank account update
        B_tp1 = update_bank_account(B_t, S_t, alpha_tm1, alpha_t, self.params.dt, self.params.epsilon, self.params.r)

        # Wealth and reward
        _, _, reward = compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t)

        return S_tp1, v_tp1, alpha_t, B_tp1, -reward


# Shared helper functions
def update_bank_account(B_t, S_t, alpha_tm1, alpha_t, dt, epsilon=0.0, r=0.0):
    B_tplus = B_t - (alpha_t - alpha_tm1) * S_t - T.abs(alpha_t - alpha_tm1) * epsilon
    return B_tplus * T.exp(r * dt)


def compute_wealth(S_t, S_tp1, B_t, B_tp1, alpha_tm1, alpha_t):
    W_t = B_t + alpha_tm1 * S_t
    W_tp1 = B_tp1 + alpha_t * S_tp1
    reward = W_tp1 - W_t
    return W_t, W_tp1, reward


def option_price(S, K, type='call', device='cpu'):
    if type == 'call':
        return T.maximum(S - K, T.zeros(1, device=device))
    elif type == 'put':
        return T.maximum(K - S, T.zeros(1, device=device))
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")


def get_final_cost(S_T, v_T, alpha_Tm1, B_T, epsilon, K):
    r = -T.abs(alpha_Tm1) * epsilon - option_price(S_T, K)
    return -r

## Write code to testing/engines/test_environments.py
from pathlib import Path
current_dir = Path().resolve() # noqa E402

import sys # noqa E402
sys.path.append(str(current_dir)) # noqa E402
print(current_dir) # noqa E402



from engines.environments import BlackScholesEnv, HestonEnv # noqa E402
from DTO.input_dtos import EnvParams
import torch as T
import numpy as np
import pytest

# Define a fixture for common environment parameters
@pytest.fixture
def env_params():
    return EnvParams(
        S0=10.0,
        K=10.0,
        v0=0.04,
        sigma=0.2,
        kappa=9,
        theta=0.0625,
        eta=1,
        T=0.08333333,
        rho=-0.5,
        mu=0.1,
        r=0.01,
        B0=0.0,
        epsilon=0.0,
        max_alpha=3,
        Ndt=31
    )

def test_black_scholes_env_reset():
    env_params_obj = EnvParams(
        S0=10.0, K=10.0, v0=0.04, sigma=0.2, kappa=9, theta=0.0625,
        eta=1, T=0.08333333, rho=-0.5, mu=0.1, r=0.01, B0=0.0,
        epsilon=0.0, max_alpha=3, Ndt=31
    )
    env = BlackScholesEnv(env_params_obj)
    S0, v0, alpha_m1, B0 = env.reset(Nsims=5)
    assert S0.shape == (5,)
    assert v0.shape == (5,)
    assert alpha_m1.shape == (5,)
    assert B0.shape == (5,)
    assert T.all(S0 == env_params_obj.S0)
    assert T.all(v0 == env_params_obj.v0)
    assert T.all(B0 == env_params_obj.B0)
    assert T.all(alpha_m1 == 0)

def test_black_scholes_env_step():
    env_params_obj = EnvParams(
        S0=10.0, K=10.0, v0=0.04, sigma=0.2, kappa=9, theta=0.0625,
        eta=1, T=0.08333333, rho=-0.5, mu=0.1, r=0.01, B0=0.0,
        epsilon=0.0, max_alpha=3, Ndt=31
    )
    env = BlackScholesEnv(env_params_obj)
    S0, v0, alpha_m1, B0 = env.reset(Nsims=5)
    alpha_t = T.tensor([1.0, -1.0, 0.5, -0.5, 0.0], device=env.device)
    S_tp1, v_tp1, alpha_t, B_tp1, cost = env.step(S0, v0, alpha_m1, B0, alpha_t)
    assert S_tp1.shape == (5,)
    assert v_tp1.shape == (5,)
    assert B_tp1.shape == (5,)
    assert cost.shape == (5,)
    assert T.all(v_tp1 == v0)  # In Black-Scholes, volatility is constant

def test_heston_env_reset():
    env_params_obj = EnvParams(
        S0=10.0, K=10.0, v0=0.04, sigma=0.2, kappa=9, theta=0.0625,
        eta=1, T=0.08333333, rho=-0.5, mu=0.1, r=0.01, B0=0.0,
        epsilon=0.0, max_alpha=3, Ndt=31
    )
    env = HestonEnv(env_params_obj)
    S0, v0, alpha_m1, B0 = env.reset(Nsims=5)
    assert S0.shape == (5,)
    assert v0.shape == (5,)
    assert alpha_m1.shape == (5,)
    assert B0.shape == (5,)
    assert T.all(S0 == env_params_obj.S0)
    assert T.all(v0 == env_params_obj.v0)
    assert T.all(B0 == env_params_obj.B0)
    assert T.all(alpha_m1 == 0)

def test_heston_env_step():
    env_params_obj = EnvParams(
        S0=10.0, K=10.0, v0=0.04, sigma=0.2, kappa=9, theta=0.0625,
        eta=1, T=0.08333333, rho=-0.5, mu=0.1, r=0.01, B0=0.0,
        epsilon=0.0, max_alpha=3, Ndt=31
    )
    env = HestonEnv(env_params_obj)
    S0, v0, alpha_m1, B0 = env.reset(Nsims=5)
    alpha_t = T.tensor([1.0, -1.0, 0.5, -0.5, 0.0], device=env.device)
    S_tp1, v_tp1, alpha_t, B_tp1, cost = env.step(S0, v0, alpha_m1, B0, alpha_t)
    assert S_tp1.shape == (5,)
    assert v_tp1.shape == (5,)
    assert B_tp1.shape == (5,)
    assert cost.shape == (5,)
    assert T.all(v_tp1 >= 0)  # Volatility should remain non-negative

if __name__ == "__main__":
    # Create env_params for standalone execution
    params = EnvParams(
        S0=10.0,
        K=10.0,
        v0=0.04,
        sigma=0.2,
        kappa=9,
        theta=0.0625,
        eta=1,
        T=0.08333333,
        rho=-0.5,
        mu=0.1,
        r=0.01,
        B0=0.0,
        epsilon=0.0,
        max_alpha=3,
        Ndt=31
    )
    
    test_black_scholes_env_reset(params)
    test_black_scholes_env_step(params)
    test_heston_env_reset(params)
    test_heston_env_step(params)  
    print("All tests passed.")
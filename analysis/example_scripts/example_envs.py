from pathlib import Path
import sys
import torch as T

project_root = str(Path().resolve())
print(f">>>Project root directory: {project_root}")
sys.path.append(str(project_root))

from engines.environments import BlackScholesEnv, HestonEnv # noqa: E402
from dto.input_dtos import EnvParams # noqa: E402


def create_debug_BS_env() -> BlackScholesEnv:
    """Create a debug Black-Scholes environment with fixed parameters."""
    params = EnvParams(
        S0=100.0,
        v0=0.0,
        B0=100.0,
        T=1.0,
        Ndt=10,
        sigma=0.2,
        r=0.05,
        max_alpha=1.0,
        K=100.0,
        kappa=0.0,
        theta=0.0,
        eta=0.0,
        rho=0.0,
        mu=0.05,
        epsilon=0.01,
    )
    env = BlackScholesEnv(params)
    return env

def create_debug_Heston_env() -> HestonEnv:
    """Create a debug Heston environment with fixed parameters."""
    params = EnvParams(
        S0=100.0,
        v0=0.04,
        B0=100.0,
        T=1.0,
        Ndt=10,
        sigma=0.2,
        r=0.05,
        max_alpha=1.0,
        K=100.0,
        kappa=2.0,
        theta=0.04,
        eta=0.1,
        rho=-0.7,
        mu=0.05,
        epsilon=0.01,
    )
    env = HestonEnv(params)
    return env


if __name__ == "__main__":
    debug_env = create_debug_BS_env()
    print(">>> Debug Black-Scholes Environment created with parameters:")
    for key, value in debug_env.params.__dict__.items():
        print(f"{key}: {value}")

    intial_state = debug_env.reset(Nsims=5)
    print(">>> Initial state after reset:\n", intial_state)

    step = debug_env.step(
        S_t=intial_state[0], 
        v_t=intial_state[1], 
        alpha_tm1=intial_state[2],
        B_t=intial_state[3], 
        alpha_t=0.5*T.ones(5))
    print(">>> State after one step:\n", step)

    random_state = debug_env.random_reset(time=0.5, Nsims=5)
    print(">>> Random state at time 0.5:\n", random_state)

    debug_env.graph_full_rollout(Nsims=5)
    debug_env.graph_random_reset(time=0.5, Nsims=5)
    debug_env.graph_random_resets_across_times(times=[0.1,.2,.3,.4,0.5,.6,.7,.8], Nsims=5)

    debug_env_heston = create_debug_Heston_env()
    print(">>> Debug Heston Environment created with parameters:")
    for key, value in debug_env_heston.params.__dict__.items():
        print(f"{key}: {value}")
    intial_state_heston = debug_env_heston.reset(Nsims=5)
    print(">>> Initial state of Heston env after reset:\n", intial_state_heston)
    step_heston = debug_env_heston.step(
        S_t=intial_state_heston[0], 
        v_t=intial_state_heston[1], 
        alpha_tm1=intial_state_heston[2],
        B_t=intial_state_heston[3], 
        alpha_t=0.5*T.ones(5))
    print(">>> State of Heston env after one step:\n", step_heston)
    random_state_heston = debug_env_heston.random_reset(time=0.5, Nsims=5)
    print(">>> Random state of Heston env at time 0.5:\n", random_state_heston)

    debug_env_heston.graph_full_rollout(Nsims=5)
    debug_env_heston.graph_random_reset(time=0.5, Nsims=5)
    debug_env_heston.graph_random_resets_across_times(times=[0.1,.2,.3,.4,0.5,.6,.7,.8], Nsims=5)
    

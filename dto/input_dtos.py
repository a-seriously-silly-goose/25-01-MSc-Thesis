from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvParams:
    S0: float
    K: float
    v0: float
    sigma: float
    kappa: float
    theta: float
    eta: float
    T: float
    rho: float
    mu: float
    r: float
    B0: float
    epsilon: float
    max_alpha: int
    Ndt: int


@dataclass
class AlgoParams:
    Ntrajectories: int
    Mtransitions: int
    Nepochs: int
    Nepochs_BC: int
    gamma: float
    Nepochs_V_init: int
    Nepochs_V: int
    lr_V: float
    batch_V: int
    hidden_V: int
    layers_V: int
    Nepochs_pi: int
    lr_pi: float
    batch_pi: int
    hidden_pi: int
    layers_pi: int
    seed: Optional[int]
    clip_epsilon: float
    lambda_entropy: float
    gae_lambda: float
    entropy_coef: float
    n_critic_updates: int
    n_actor_updates: int


@dataclass
class RiskParams:
    method: str


@dataclass
class RunParams:
    # Add fields when run-specific parameters are defined
    pass

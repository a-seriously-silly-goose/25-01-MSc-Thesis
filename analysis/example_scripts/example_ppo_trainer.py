import torch
import numpy as np

from pathlib import Path
import os
import sys
import yaml
path = Path(os.getcwd())

sys.path.append(str(path))
print("\n>>>", str(path), "\n")

from dto.config_loader import load_config
from engines.environments import HestonEnv
from engines.agents.models import PolicyNN, CriticNN
from engines.agents.actor_critic_ppo import ActorCriticPPO  # Adjust import as needed

def make_dummy_data(batch_size, obs_dim, act_dim):
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randint(0, act_dim, (batch_size,))
    rewards = torch.randn(batch_size)
    dones = torch.zeros(batch_size)
    values = torch.randn(batch_size)
    log_probs = torch.randn(batch_size)
    advantages = torch.randn(batch_size)
    returns = torch.randn(batch_size)
    return {
        'obs': obs,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'values': values,
        'log_probs': log_probs,
        'advantages': advantages,
        'returns': returns
    }

def main():
    # Load params from YAML for debug (like in example_beh_clone)

    # Define environment parameters
    config_path = "runs/hyperparameters.yml"
    params_key = "v_debug"
    env_params, algo_params, _ = load_config(path=config_path, version=params_key)

    # Initialize the Black-Scholes environment
    env = HestonEnv(env_params)

    # Create a simple policy network structure for behavior cloning
    policy_net = PolicyNN(input_dto=algo_params, env=env)
    value_net = CriticNN(input_dto=algo_params, env=env)

    ppo = ActorCriticPPO(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        algoParams=algo_params,
        hyperparameters_version=params_key,
        root_repo=str(path)
    )


    # Debug train_ppo
    print("Starting train_ppo debug...")
    loss_info = ppo.train_ppo(K_PPO=100)
    print("train_ppo output:", loss_info)

if __name__ == "__main__":
    main()
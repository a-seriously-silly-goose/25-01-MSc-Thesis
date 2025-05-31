import os
import torch as T
from models import PolicyApprox, ValueApprox  # Assuming these are defined
from envs import BlackScholesEnv
from agents.actor_critic_ppo import ActorCriticPPO  # Your PPO implementation


print(os.getcwd() + "\n" + __file__ + "\n" + os.path.dirname(__file__) + "\n")
# Set up hyperparameters
hyperparameters = {
    "envParams": {
        "S0": 10,  # initial stock price
        "B0": 0,  # initial bank account
        "K": 10,  # strike price
        "T": 0.08333333,  # time to maturity
        "mu": 0.00,  # drift in GBM
        "sigma": 0.2,  # volatility in GBM
        "r": 0.01,  # risk-free rate
        "epsilon": 0,  # transaction cost
        "alpha0": 0,  # initial position
        "max_alpha": 8,  # maximum position
        "Ndt": 31,  # number of time steps
        "kappa": 9,
        "theta": 0.1,
        "eta": 0.1,
        "v0": 0.2,  # initial volatility
    },
    "algoParams": {
        "Ntrajectories": 100,  # trajectories per epoch
        "Mtransitions": 31,  # transitions per trajectory (matches Ndt)
        "Nepochs": 100,  # training epochs
        "gamma": 0.99,  # discount factor
        "seed": 42,  # random seed
        "clip_epsilon": 0.2,  # PPO clip parameter
        "n_actor_updates": 4,  # actor updates per epoch
        "n_critic_updates": 4,  # critic updates per epoch
        "gae_lambda": 0.95,  # GAE lambda
        "entropy_coef": 0.01,  # entropy bonus coefficient
        "lr_actor": 0.0003,  # actor learning rate
        "lr_critic": 0.001,  # critic learning rate
        "hidden_size": 64,  # hidden units per layer
        "policy_layers": 3,  # policy network depth
        "value_layers": 3,  # value network depth
        "risk_measure": "CVAR",  # CVAR/MeanVar/etc.
        "cvar_alpha": 0.95,  # CVAR confidence level
    },
    "hyperparameters_version": "vCJ21",
    "log_file": "ppo_training_log.txt",
    "repo": "./ppo_runs",
    "model_save_name": "ppo_hedging_agent",
}

# Setup directories and logging
os.makedirs(hyperparameters["repo"], exist_ok=True)
log_path = os.path.join(hyperparameters["repo"], hyperparameters["log_file"])

# Initialize environment
env = BlackScholesEnv(hyperparameters["envParams"])

# Initialize policy and value networks
policy_net = PolicyApprox(
    input_size=3,  # [price, holdings, time]
    env=env,
    n_layers=hyperparameters["algoParams"]["policy_layers"],
    hidden_size=hyperparameters["algoParams"]["hidden_size"],
    learn_rate=hyperparameters["algoParams"]["lr_actor"],
)

value_net = ValueApprox(
    input_size=3,  # Same state input
    env=env,
    n_layers=hyperparameters["algoParams"]["value_layers"],
    hidden_size=hyperparameters["algoParams"]["hidden_size"],
    learn_rate=hyperparameters["algoParams"]["lr_critic"],
)

# Initialize PPO agent
ppo_agent = ActorCriticPPO(
    repo=hyperparameters["repo"],
    method="PPO",
    env=env,
    policy=policy_net,
    V=value_net,
    risk_measure=hyperparameters["algoParams"]["risk_measure"],
    hyperparameter_set=hyperparameters["hyperparameters_version"],
    LOG_FILE=log_path,
)

# Configure risk measure parameters if needed
# if hyperparameters["algoParams"]["risk_measure"] == "CVAR":
#     ppo_agent.set_cvar_alpha(hyperparameters["algoParams"]["cvar_alpha"])

# Training loop
print("Starting PPO training...")
for epoch in range(hyperparameters["algoParams"]["Nepochs"]):
    # Update policy with PPO
    ppo_agent.update_policy(
        Ntrajectories=hyperparameters["algoParams"]["Ntrajectories"],
        Mtransitions=hyperparameters["algoParams"]["Mtransitions"],
        batch_size=hyperparameters["algoParams"]["Ntrajectories"],  # Full batch
        Nepochs=1,  # Single epoch per update (handled internally)
        rng_seed=(
            hyperparameters["algoParams"]["seed"] + epoch
            if hyperparameters["algoParams"]["seed"]
            else None
        ),
    )

    # Save model checkpoint
    if (epoch + 1) % 10 == 0:
        model_path = os.path.join(
            hyperparameters["repo"],
            f"{hyperparameters['model_save_name']}_epoch{epoch+1}.pt",
        )
        T.save(
            {
                "policy_state_dict": ppo_agent.policy.state_dict(),
                "value_state_dict": ppo_agent.V.state_dict(),
                "optimizer_policy": ppo_agent.policy.optimizer.state_dict(),
                "optimizer_value": ppo_agent.V.optimizer.state_dict(),
            },
            model_path,
        )
        print(f"Saved model checkpoint at epoch {epoch+1}")

# Save final model
final_model_path = os.path.join(
    hyperparameters["repo"], f"{hyperparameters['model_save_name']}_final.pt"
)
T.save(ppo_agent.policy.state_dict(), final_model_path)

print("PPO training completed successfully!")
print(f"Final model saved to: {final_model_path}")
print(f"Training log saved to: {log_path}")

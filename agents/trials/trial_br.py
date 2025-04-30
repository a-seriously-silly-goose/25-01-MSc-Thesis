import os
import torch as T
from models import PolicyApprox
from envs import BlackScholesEnv
from agents.actor_delta_hedge import DeltaHedgeActor
from agents.behaviour_replication import (
    BehaviorReplicationAgent,
)  # Import the replication agent

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
        "v0": 0.2,  # initial v
    },
    "algoParams": {
        "Ntrajectories": 50,  # number of generated trajectories
        "Mtransitions": 50,  # number of additional transitions for each state
        "Nepochs": 40,  # number of epochs of the whole algorithm
        "gamma": 1,  # discount factor
        "seed": 42,  # (int or None) set seed for replication purposes
        "Nepochs_V_init": 150,  # number of epochs for the estimation of V during the first epoch
        "Nepochs_V": 300,  # number of epochs for the estimation of V
        "lr_V": 0.0005,  # learning rate of the neural net associated with V
        "batch_V": 20,  # number of trajectories for each mini-batch in estimating V
        "hidden_V": 16,  # number of hidden nodes in the neural net associated with V
        "layers_V": 4,  # number of layers in the neural net associated with V
        "Nepochs_pi": 20,  # number of epoch for the update of pi
        "lr_pi": 0.0005,  # learning rate of the neural net associated with pi
        "batch_pi": 20,  # number of trajectories for each mini-batch when updating pi
        "hidden_pi": 16,  # number of hidden nodes in the neural net associated with pi
        "layers_pi": 3,  # number of layers in the neural net associated with pi
    },
    "hyperparameters_version": "d002.000.001",  # Version for logging/tracking
    "log_file": "replication_log.txt",  # Log file path
    "repo": "./runs",  # Directory for saving model and logs
}

# Create necessary directories
os.makedirs(hyperparameters["repo"], exist_ok=True)

# Set up parameters
envParams = hyperparameters["envParams"]
algoParams = hyperparameters["algoParams"]
hyperparameters_version = hyperparameters["hyperparameters_version"]
log_file = os.path.join(hyperparameters["repo"], hyperparameters["log_file"])
repo = hyperparameters["repo"]

# Initialize the environment
env = BlackScholesEnv(envParams)

# Initialize the Delta Hedge Actor
delta_hedge_actor = DeltaHedgeActor(
    repo, "DH_actor", env, hyperparameters_version, log_file
)

# Initialize policy structure (same as PG/PPO)
policy_structure = PolicyApprox(
    input_size=3,  # Features: S, alpha, time
    env=env,
    n_layers=algoParams["layers_pi"],
    hidden_size=algoParams["hidden_pi"],
    learn_rate=algoParams["lr_pi"],
)

# Initialize the Behavior Replication Agent
replication_agent = BehaviorReplicationAgent(
    repo=repo,
    method="Behavior_Repl",
    env=env,
    policy_structure=policy_structure,
    V_structure=None,  # Not needed for replication
    hyperparameters_version=hyperparameters_version,
    LOG_FILE=log_file,
)

# Train the replication agent to imitate the delta hedge act or
replication_agent.load_policy(replication_agent.PI_MODEL_FILE)

print("Starting behavior replication...")
replication_agent.replicate_delta_hedging(delta_hedge_actor)

# Save the trained policy
replication_agent.save_policy()

print("Behavior replication completed successfully!")

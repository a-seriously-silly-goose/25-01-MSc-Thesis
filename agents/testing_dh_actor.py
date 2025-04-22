from agents.actor_delta_hedge import DeltaHedgeActor
from envs import BlackScholesEnv
import torch as T

# personal files
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import BlackScholesEnv
from agents.actor_critic_pg import ActorCriticPG

# misc
import os
import argparse
import yaml


def setup_experiment(hyperparameters_version, is_training=False, preload=False):
    import yaml
    import os
    import torch as T
    from datetime import datetime

    # Paths & Device
    RUNS_DIR = "runs"
    os.makedirs(RUNS_DIR, exist_ok=True)
    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    root_path = os.path.join(os.path.dirname(os.getcwd()))

    # Load hyperparameters
    with open(os.path.join(root_path, "hyperparameters.yml"), "r") as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameters_version]

    envParams = hyperparameters["envParams"]
    algoParams = hyperparameters["algoParams"]
    riskParams = hyperparameters["riskParams"]
    runParams = hyperparameters["runParams"]
    repo_name = hyperparameters_version

    # Log message
    log_message = (
        f"*** Name of the repository:  {repo_name} ***\n"
        f"*** Environment parameters:  {envParams} ***\n"
        f"*** Algorithm parameters:  {algoParams} ***\n"
        f"*** Risk measures parameters:  {riskParams} ***\n"
        f"*** Run parameters:  {runParams} ***\n"
        f"*** Risk measures:  ['mean'] ***\n"
        f"*** alpha_cvar:  {[0.2]}"
    )
    print(log_message)

    # Assert Feller condition
    assert (
        2 * envParams["kappa"] * envParams["theta"] > envParams["eta"] ** 2
    ), "Feller condition is not satisfied."

    # Create repository
    repo = os.path.join(RUNS_DIR, repo_name)
    os.makedirs(repo, exist_ok=True)

    # Save log file
    LOG_FILE = os.path.join(repo, f"{hyperparameters_version}.log")
    with open(LOG_FILE, "w") as file:
        file.write(log_message + "\n")

    return {
        "device": device,
        "envParams": envParams,
        "algoParams": algoParams,
        "riskParams": riskParams,
        "runParams": runParams,
        "repo": repo,
        "log_file": LOG_FILE,
        "hyperparameters_version": hyperparameters_version,
        "is_training": is_training,
        "preload": preload,
    }


config = setup_experiment(
    hyperparameters_version="d002.000.001", is_training=True, preload=False
)

# Now you can pass `config` to build the env, agent, etc.


env = BlackScholesEnv(config["envParams"])
DH_agent = DeltaHedgeActor(
    os.getcwd(),
    "DH_actor",
    env,
    config["hyperparameters_version"],
    config["log_file"],
)


test_trajectories = DH_agent.sim_trajectories(1, 5)


DH_agent.plot_current_policy2()
DH_agent.plot_delta_vs_time()
DH_agent.plot_delta_vs_stock_price()

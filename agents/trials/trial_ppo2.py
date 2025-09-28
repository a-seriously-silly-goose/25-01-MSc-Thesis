import os
import torch as T
from models import PolicyApprox, ValueApprox  # Assuming these are defined
from envs import BlackScholesEnv
from agents.actor_critic_ppo import ActorCriticPPO

"""
Main -- Financial Hedging Problem
Value function & policy represented by a single ANN
Value function is learned from the current policy
"""

# numpy
# plotting
from scipy.stats import gaussian_kde

# pytorch
import torch as T

# personal files
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import BlackScholesEnv as HedgingEnv
from agents.actor_critic_pg import ActorCriticPG

# misc
import time
import os
from datetime import datetime
import argparse
import yaml

"""
Parameters
"""

# Create directory for storing runs
DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# matplotlib.use('Agg')
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Parse command line inputs
parser = argparse.ArgumentParser(description="Train or test model.")
parser.add_argument("hyperparameters", help="")
parser.add_argument("--train", help="Training mode", action="store_true")
parser.add_argument(
    "--preload", help="Should the existing model be preloaded", action="store_true"
)
args = parser.parse_args(args=["vCJ21", "--train"])
# args = parser.parse_args()

preload = args.preload
hyperparameters_version = args.hyperparameters
is_training = args.train
print("Hyperparameter set:", hyperparameters_version)
with open(
    "/Users/simeon/Documents/GitHub/03 University/25 01 MSc Thesis/hyperparameters.yml",
    "r",
) as file:
    all_hyperparameter_sets = yaml.safe_load(file)
    hyperparameters = all_hyperparameter_sets[hyperparameters_version]

envParams = hyperparameters["envParams"]
algoParams = hyperparameters["algoParams"]
riskParams = hyperparameters["riskParams"]
runParams = hyperparameters["runParams"]
repo_name = hyperparameters_version

# risk measures used
rm_list = ["mean"]  # 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
alpha_cvar = [0.2]  # threshold for the conditional value-at-risk
kappa_semidev = [-99]  # coefficient for the mean semideviation
r_semidev = [-99]  # exponent of the mean-semideviation

print_progress = 200  # number of epochs before printing the time/loss
plot_progress = 20  # number of epochs before plotting the policy/value function
save_progress = 20  # number of epochs before saving the policy/value function ANNs

"""
End of Parameters
"""
log_message = (
    f"*** Name of the repository:  {repo_name} ***\n"
    f"*** Environment parameters:  {envParams} ***\n"
    f"*** Algorithm parameters:  {algoParams} ***\n"
    f"*** Risk measures parameters:  {riskParams} ***\n"
    f"*** Run parameters:  {runParams} ***\n"
    f"*** Risk measures:  {rm_list} ***\n"
    f"*** alpha_cvar:  {alpha_cvar}"
)

print(log_message)
# print all parameters for reproducibility purposes
print("\n*** Name of the repository: ", repo_name, " ***\n")


# Feller condition
assert (
    2 * envParams["kappa"] * envParams["theta"] > envParams["eta"] ** 2
), "Feller condition is not satisfied."

# Create repository for storing result
repo = os.path.join(RUNS_DIR, repo_name)
if not os.path.exists(repo):
    os.makedirs(repo)
LOG_FILE = os.path.join(repo, f"{hyperparameters_version}.log")
with open(LOG_FILE, "w") as file:
    file.write(log_message + "\n")

# loop for all risk measures
for idx_method, method in enumerate(rm_list):
    # print progress
    print("\n*** Method = ", method, " ***\n")
    start_time = time.time()

    # create the environment and risk measure objects
    env = HedgingEnv(envParams)
    risk_measure = RiskMeasure(
        Type=method,
        alpha=alpha_cvar[idx_method],
        kappa=kappa_semidev[idx_method],
        r=r_semidev[idx_method],
    )

    # create repositories
    if method == "CVaR":
        method = method + str(round(alpha_cvar[idx_method], 3))
    if method == "semi-dev":
        method = method + str(round(kappa_semidev[idx_method], 3))
    if method == "mean-CVaR":
        method = (
            "mean"
            + str(round(kappa_semidev[idx_method], 3))
            + "-CVaR"
            + str(round(alpha_cvar[idx_method], 3))
        )
    if method == "CVaR-penalized":
        method = (
            "CVaR"
            + str(round(alpha_cvar[idx_method], 3))
            + "-pen"
            + str(round(kappa_semidev[idx_method], 3))
        )

    # utils.directory(repo + '/' + method)
    # utils.directory(repo + '/' + method + '/evolution')

    # create policy & value function objects
    # single neural network; (price x hedge x bank account x time)
    policy = PolicyApprox(
        4,
        env,
        n_layers=algoParams["layers_pi"],
        hidden_size=algoParams["hidden_pi"],
        learn_rate=algoParams["lr_pi"],
    )
    value_function = ValueApprox(
        4,
        env,
        n_layers=algoParams["layers_V"],
        hidden_size=algoParams["hidden_V"],
        learn_rate=algoParams["lr_V"],
    )

    # initialize the actor-critic algorithm
    actor_critic = ActorCriticPPO(
        env=env,
        policy_net=policy,
        value_net=value_function,
        hyperparameter_set="vCJ21",
        log_file=LOG_FILE,
    )

print(os.getcwd() + "\n" + __file__ + "\n" + os.path.dirname(__file__) + "\n")

# Setup directories and logging
# os.makedirs(hyperparameters["repo"], exist_ok=True)
# log_path = os.path.join(hyperparameters["repo"], hyperparameters["log_file"])

# Training loop
for iteration in range(100):
    # Behavioral cloning could be done here first
    actor_critic.update(num_trajectories=2048)  # Collect 2048 trajectories per update

    # Save models periodically
    if iteration % 10 == 0:
        actor_critic.save_models(f"checkpoints/iteration_{iteration}")

"""
Main -- Financial Hedging Problem
Value function & policy represented by a single ANN
Value function is learned from the current policy
"""
# numpy
import numpy as np
import numpy.matlib
# plotting
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
# pytorch
import torch as T
import torch.optim as optim
# personal files
import utils
from models import PolicyApprox, ValueApprox
from risk_measure import RiskMeasure
from envs import HedgingEnv
from actor_critic import ActorCriticPG
# misc
import time
import os
import pdb # use with set_trace() for the debugger
from datetime import datetime
import argparse
import yaml

"""
Parameters
"""

# Create directory for storing runs
DATE_FORMAT = '%m-%d %H:%M:%S'
RUNS_DIR = 'runs'
os.makedirs(RUNS_DIR, exist_ok=True)

#matplotlib.use('Agg')
device = T.device("cuda" if T.cuda.is_available() else "cpu")

# Parse command line inputs
parser = argparse.ArgumentParser(description='Train or test model.')
parser.add_argument('hyperparameters', help='')
parser.add_argument('--train', help='Training mode', action='store_true')
parser.add_argument('--preload', help='Should the existing model be preloaded', action='store_true')
args = parser.parse_args()

preload= args.preload
hyperparameters_version = args.hyperparameters
is_training = args.train
print('Hyperparameter set:', hyperparameters_version)
with open('hyperparameters.yml','r') as file:
    all_hyperparameter_sets = yaml.safe_load(file)
    hyperparameters = all_hyperparameter_sets[hyperparameters_version]

envParams = hyperparameters['envParams']
algoParams = hyperparameters['algoParams']
riskParams = hyperparameters['riskParams']
runParams = hyperparameters['runParams']
repo_name = hyperparameters_version

# risk measures used
rm_list = ['mean'] # 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
alpha_cvar = [ 0.2] # threshold for the conditional value-at-risk
kappa_semidev = [ -99] # coefficient for the mean semideviation
r_semidev = [-99] # exponent of the mean-semideviation

print_progress = 200 # number of epochs before printing the time/loss
plot_progress = 50 # number of epochs before plotting the policy/value function
save_progress = 100 # number of epochs before saving the policy/value function ANNs

"""
End of Parameters
"""
log_message =f'*** Name of the repository:  {repo_name} ***\n'\
            f'*** Environment parameters:  {envParams} ***\n'\
            f'*** Algorithm parameters:  {algoParams} ***\n'\
            f'*** Risk measures parameters:  {riskParams} ***\n'\
            f'*** Run parameters:  {runParams} ***\n'\
            f'*** Risk measures:  {rm_list} ***\n'\
            f'*** alpha_cvar:  {alpha_cvar}'

print(log_message)
# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')


# Feller condition
assert (2*envParams["kappa"]*envParams["theta"] > envParams["eta"]**2), "Feller condition is not satisfied."

# Create repository for storing result
repo = os.path.join(RUNS_DIR, repo_name)
if not os.path.exists(repo):
    os.makedirs(repo)
LOG_FILE = os.path.join(repo, f'{hyperparameters_version}.log')
with open(LOG_FILE,'w') as file:
    file.write(log_message + '\n')

# loop for all risk measures
for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the environment and risk measure objects
    env = HedgingEnv(envParams)
    risk_measure = RiskMeasure(Type=method,
                                alpha=alpha_cvar[idx_method],
                                kappa=kappa_semidev[idx_method],
                                r=r_semidev[idx_method])

    # create repositories
    if(method == 'CVaR'):
        method = method + str( round(alpha_cvar[idx_method],3) )
    if(method == 'semi-dev'):
        method = method + str( round(kappa_semidev[idx_method],3) )
    if(method == 'mean-CVaR'):
        method = 'mean' + str(round(kappa_semidev[idx_method],3)) \
                    + '-CVaR' + str(round(alpha_cvar[idx_method],3))
    if(method == 'CVaR-penalized'):
        method = 'CVaR' + str(round(alpha_cvar[idx_method],3)) \
                    + '-pen' + str(round(kappa_semidev[idx_method],3))

    #utils.directory(repo + '/' + method)
    #utils.directory(repo + '/' + method + '/evolution')

    # create policy & value function objects
    # single neural network; (price x hedge x bank account x time)
    policy = PolicyApprox(3, env,
                            n_layers=algoParams["layers_pi"],
                            hidden_size=algoParams["hidden_pi"],
                            learn_rate=algoParams["lr_pi"])
    value_function = ValueApprox(3, env,
                            n_layers=algoParams["layers_V"],
                            hidden_size=algoParams["hidden_V"],
                            learn_rate=algoParams["lr_V"])
    
    # initialize the actor-critic algorithm
    actor_critic = ActorCriticPG(repo=repo,
                                    method = method,
                                    env=env,
                                    policy=policy,
                                    V=value_function,
                                    risk_measure=risk_measure,
                                    hyperparameter_set=hyperparameters_version,
                                    LOG_FILE=LOG_FILE)

    if is_training:
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)

        start_time = time.time()
        with open(actor_critic.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

    if preload:
        # load the weights of the pre-trained model
        actor_critic.policy.load_state_dict(T.load(data_repo + '/' + method + '/policy_model.pt'))
        actor_critic.V.load_state_dict(T.load(data_repo + '/' + method + '/V_model.pt'))

    ## TRAINING PHASE
    # first estimate of the value function
    actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                Mtransitions=algoParams["Mtransitions"],
                                batch_size=algoParams["batch_V"],
                                Nepochs=algoParams["Nepochs_V_init"],
                                rng_seed=algoParams["seed"])

    # plot current policy
    actor_critic.plot_current_V()
    actor_critic.plot_current_policy()

    for epoch in range(algoParams["Nepochs"]):
        # estimate the value function of the current policy
        actor_critic.estimate_V(Ntrajectories=algoParams["Ntrajectories"],
                                    Mtransitions=algoParams["Mtransitions"],
                                    batch_size=algoParams["batch_V"],
                                    Nepochs=algoParams["Nepochs_V"],
                                    rng_seed=algoParams["seed"])
        
        # update the policy by policy gradient
        actor_critic.update_policy(Ntrajectories=algoParams["Ntrajectories"],
                                    Mtransitions=algoParams["Mtransitions"],
                                    batch_size=algoParams["batch_pi"],
                                    Nepochs=algoParams["Nepochs_pi"],
                                    rng_seed=algoParams["seed"])

        # print progress
        if epoch % print_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            log_message= f"*** Epoch =  {str(epoch)} completed, Duration = {(time.time() - start_time): .3f} secs ***"
            start_time = time.time()
            print(log_message)
            with open(actor_critic.LOG_FILE, 'a') as file:
                file.write(log_message + '\n')

        # plot current policy
        if epoch % plot_progress == 0 or epoch == algoParams["Nepochs"] - 1:
            actor_critic.plot_current_V()
            actor_critic.plot_current_policy()

        # save progress
        if epoch % save_progress == 0:
            now = datetime.now()
            # save the neural network
            T.save(actor_critic.policy.state_dict(), actor_critic.PI_MODEL_FILE)
            T.save(actor_critic.V.state_dict(), actor_critic.V_MODEL_FILE)

    # save the neural network
    T.save(actor_critic.policy.state_dict(), actor_critic.PI_MODEL_FILE)
    T.save(actor_critic.V.state_dict(), actor_critic.V_MODEL_FILE)
    # to load the model, M = ModelClass(*args, **kwargs); M.load_state_dict(T.load(PATH))

    # print progress
    log_message ='*** Training phase completed! ***'
    print(log_message)
    with open(actor_critic.LOG_FILE, 'a') as file:
        file.write(log_message+'\n')
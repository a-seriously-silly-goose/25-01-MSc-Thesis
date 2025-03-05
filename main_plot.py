"""
Plots -- Cliff Walking Problem
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
from risk_measures import RiskMeasure
from env import BlackScholesEnv
from agent import ActorCriticPG
# misc
import time
import os
import pdb # use with set_trace() for the debugger
import argparse
import yaml
"""
Parameters
"""

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
rm_list = ['CVaR'] # 'mean' | 'CVaR' | 'semi-dev' | 'CVaR-penalized' | 'mean-CVaR'
alpha_cvar = [ 0.2] # threshold for the conditional value-at-risk
kappa_semidev = [ -99] # coefficient for the mean semideviation
r_semidev = [-99] # exponent of the mean-semideviation


print_progress = 200 # number of epochs before printing the time/loss
plot_progress = 50 # number of epochs before plotting the policy/value function
save_progress = 100 # number of epochs before saving the policy/value function ANNs

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
log_message =f'*** Name of the repository:  {repo_name} ***\n'\
            f'*** Environment parameters:  {envParams} ***\n'\
            f'*** Algorithm parameters:  {algoParams} ***\n'\
            f'*** Risk measures parameters:  {riskParams} ***\n'\
            f'*** Run parameters:  {runParams} ***\n'\
            f'*** Risk measures:  {rm_list} ***\n'\
            f'*** alpha_cvar:  {alpha_cvar}'

print(log_message)

# Ensure learning rates are converted to float
algoParams["lr_pi"] = float(algoParams["lr_pi"])
algoParams["lr_V"] = float(algoParams["lr_V"])

# Check if preloading is required
#preload = runParams.get("preload", False)
#data_repo = runParams.get("data_repo", "")

# Create repository for storing result
repo = os.path.join('runs', repo_name)
if not os.path.exists(repo):
    os.makedirs(repo)
LOG_FILE = os.path.join(repo, f'{hyperparameters_version}_plot.log')


seed = 4321 # set seed for replication purposes

# testing phase parameters
Nsimulations = 30000 # number of simulations following the optimal strategy

# font sizes for figures
plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=20)

"""
End of Parameters
"""

# print all parameters for reproducibility purposes
print('\n*** Name of the repository: ', repo_name, ' ***\n')

utils.directory(repo)

costs = np.zeros((Nsimulations, envParams["Ndt"], len(rm_list))) # matrix to store all testing trajectories
finalprice = np.zeros((Nsimulations, len(rm_list))) # matrix to store all testing trajectories

for idx_method, method in enumerate(rm_list):
    # print progress
    print('\n*** Method = ', method, ' ***\n')
    start_time = time.time()

    # create the (temporary) environment and risk measure objects
    env = BlackScholesEnv(envParams)
    risk_measure = RiskMeasure(Type='mean')

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
                                    env=env,
                                    policy=policy,
                                    V=value_function,
                                    risk_measure=risk_measure,
                                    hyperparameter_set=hyperparameters_version,
                                    LOG_FILE=LOG_FILE)

    # load the trained model
    actor_critic.policy.load_state_dict(T.load(repo + '/Pi '+ hyperparameters_version+'.pt',map_location=T.device('cpu'), weights_only=True))
    actor_critic.V.load_state_dict(T.load(repo + '/V '+ hyperparameters_version+'.pt',map_location=T.device('cpu'), weights_only=True))

    # print progress
    print('*** Training phase completed! ***')

    ## TESTING PHASE
    # set seed for reproducibility purposes
    T.manual_seed(seed)
    np.random.seed(seed)

    # initialize the starting state
    S, alpha, B = env.reset(Nsimulations)
    
    for timestep in env.spaces["t_space"][:-1]:
        # simulate transitions according to the policy
        u, _ = actor_critic.select_actions(S, alpha, B, timestep*T.ones(Nsimulations), 'best')
        S, alpha, B, cost = env.step(S, alpha, B, u)

        # store costs
        costs[:,timestep,idx_method] = cost.detach().numpy()

    # get terminal reward
    costs[:,-1,idx_method] = env.settlement(S, alpha, B).detach().numpy()
    finalprice[:,idx_method] = S.detach().numpy()

    ### PLOT - policy wrt price and time
    # initialize 2D histogram
    hist2dim_pi = np.zeros([len(env.spaces["S_space"]), len(env.spaces["t_space"])-1])
    
    # fixed values for other variables
    fixed_alpha = 0.0
    fixed_B = env.params["B0"]

    for S_idx, S_val in enumerate(env.spaces["S_space"]):
        for time_idx, time_val in enumerate(env.spaces["t_space"][:-1]):
            # mean of the Gaussian policy
            hist2dim_pi[len(env.spaces["S_space"])-S_idx-1, time_idx], _ = \
                    actor_critic.select_actions(T.Tensor([S_val]),
                                                T.tensor([fixed_alpha]),
                                                T.tensor([fixed_B]),
                                                T.tensor([time_val]),
                                                False)

    # plot the 2D histogram
    plt.imshow(hist2dim_pi,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(env.spaces["t_space"]), 
                        np.max(env.spaces["t_space"]),
                        np.min(env.spaces["S_space"]),
                        np.max(env.spaces["S_space"])],
                aspect='auto',
                vmin=-env.params["max_alpha"],
                vmax=env.params["max_alpha"])

    plt.xlabel("Time")
    plt.ylabel("Price of the asset")
    plt.title("Learned Policy")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(repo + '/learnedpolicy_' + method + '_allperiods.pdf', transparent=True)
    plt.clf()

"""
# Plots & figures
"""

# plot rewards instead of costs
rewards_total = -1 * np.sum(costs, axis=1) + env.params["B0"]

# set a grid for the histogram
grid = np.linspace(np.min(rewards_total), np.max(rewards_total), 100)

### PLOT - Distribution of the terminal reward
for idx_method, method in enumerate(rm_list):
    # plot the histogram for each method
    plt.hist(x=rewards_total[:,idx_method],
            alpha=0.4,
            bins=grid,
            color=utils.colors[idx_method],
            density=True)

plt.legend(rm_list)
plt.xlabel("Terminal reward")
plt.ylabel("Density")
plt.title("Distribution of the terminal reward")

for idx_method, method in enumerate(rm_list):
    # plot gaussian KDEs
    kde = gaussian_kde(rewards_total[:,idx_method], bw_method='silverman')
    plt.plot(grid,
            kde(grid),
            color=utils.colors[idx_method],
            linewidth=1.5)
    # plot quantiles of the distributions
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.1),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)
    # plt.axvline(x=np.mean(rewards_total[:,idx_method]),
    #             linestyle='dotted',
    #             color=utils.colors[idx_method],
    #             linewidth=1.0)
    plt.axvline(x=np.quantile(rewards_total[:,idx_method],0.9),
                linestyle='dashed',
                color=utils.colors[idx_method],
                linewidth=1.0)

plt.tight_layout()
plt.savefig(repo + '/comparison_terminal_cost.pdf', transparent=True)
plt.clf()

### PLOT - Payoff at the terminal time
for idx_method, method in enumerate(rm_list):
    plt.scatter(finalprice[:,idx_method],
                -rewards_total[:,idx_method] + np.maximum( env.params["K"]-finalprice[:,idx_method] , 0),
                alpha=0.15,
                s=2,
                color=utils.mred)    
    plt.title('Terminal payoff')
    plt.xlabel("Price of the asset")
    plt.ylabel("Bank account")
    plt.tight_layout()
    plt.savefig(repo + '/payoff_' + method + '.pdf', transparent=True)
    plt.clf()

# print progress
print('*** Testing phase completed! ***')
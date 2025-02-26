import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch as T
from torch import nn
from torch.distributions import Normal

import yaml


from datetime import datetime, timedelta
import argparse
import os
import utils

import pdb

# create directory for storing runs

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class ActorCriticPG():

    def __init__(self, 
                 repo,
                 env,
                 policy,
                 V,
                 risk_measure,
                 hyperparameter_set,
                 LOG_FILE):

        with open('hyperparameters.yml','r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.policy = policy
        self.V = V
        self.risk_measure = risk_measure
        self.env = env
        self.repo = repo
        self.device = self.policy.device

        self.loss_history_policy = []
        self.loss_history_V = []
        self.loss_trail = 100
        self.loss_print = 50

        self.LOG_FILE = LOG_FILE
        self.V_MODEL_FILE = os.path.join(self.repo, f'V {hyperparameter_set}.pt')
        self.PI_MODEL_FILE = os.path.join(self.repo, f'Pi {hyperparameter_set}.pt')
        self.GRAPH_FILE1 = os.path.join(self.repo, f'{hyperparameter_set}_PLOT1.png')
        self.GRAPH_FILE2 = os.path.join(self.repo, f'{hyperparameter_set}_PLOT2.png')
        self.GRAPH_FILE3 = os.path.join(self.repo, f'{hyperparameter_set}_PLOT3.png')
        self.GRAPH_FILE4 = os.path.join(self.repo, f'{hyperparameter_set}_PLOT4.png')

    def select_actions(self,
                       S_t,
                       alpha_t,
                       B_t,
                       time_t,
                       is_action_random=False,
                       seed=None):
        
        assert S_t.shape[0] == alpha_t.shape[0], "S and v must have same shape"
        assert alpha_t.shape[0] == B_t.shape[0], "alpha and B must have same shape"
        assert B_t.shape[0] == time_t.shape[0], "B and time must have same shape"

        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)

        # observations as a formatted tensor
        obs_t = T.stack((S_t.clone(),
                        alpha_t.clone(),
                        time_t.clone()),-1)
        
        # obtain parameters of the distribution
        actions_param1, actions_param2 = self.policy(obs_t.clone()) # location- scale parameters

        # create action distributions with a Normal distribution
        actions_dist = Normal(actions_param1, actions_param2)
     
        # get action from the policy
        if is_action_random:
            actions_sample = actions_dist.rsample()  # random sample from the Normal
        else:
            actions_sample = actions_param1  # mode of the Normal
        
        
        # get actions
        u_t = T.maximum(T.ones(1, device = self.device)*-self.env.params["max_alpha"], \
                        T.minimum(T.ones(1, device = self.device)*self.env.params["max_alpha"], \
                                    actions_sample.squeeze(-1)))

        # get log-probabilities of the action
        log_prob_t = actions_dist.log_prob(actions_sample.detach()).squeeze()
        
        # verification of any problem with log_prob
        if(T.isnan(log_prob_t).any() or T.isinf(log_prob_t).any()):
            assert False, "missing or infinite values in the gradients"
        
        return u_t, log_prob_t

    # simulate trajectories from the policy
    def sim_trajectories(self,
                        Ntrajectories=100, # number of trajectories
                        Mtransitions=100, # number of transitions
                        is_action_random=False, # how to choose the actions
                        seed=None): # random seed
        
        # freeze the seed
        #pdb.set_trace()
        if seed is not None:
            T.manual_seed(seed)
            np.random.seed(seed)
        
        # initialize tables for all trajectories
        S = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        alpha = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        B = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep = T.zeros((Ntrajectories, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)

        S_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        alpha_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        B_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        timestep_tp1 = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        u_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        log_prob_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        cost_t = T.zeros((Ntrajectories, Mtransitions, self.env.params["Ndt"]), \
                                dtype=T.float, requires_grad=False, device=self.device)
        
        # simulate N whole trajectories
        for t_idx in self.env.spaces["t_space"]:
            # starting state (outer) with multiple random states
            S[:,t_idx], alpha[:,t_idx], B[:,t_idx] = \
                            self.env.random_reset(t_idx, Ntrajectories)
            timestep[:,t_idx] = t_idx

            # get actions from the policy (inner)
            u_t[:,:,t_idx], log_prob_t[:,:,t_idx] = \
                            self.select_actions(S[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                alpha[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                B[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                timestep[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                                is_action_random)

            # simulate transitions (inner): multiple actions
            S_tp1[:,:,t_idx], alpha_tp1[:,:,t_idx], \
            B_tp1[:,:,t_idx], cost_t[:,:,t_idx] = \
                            self.env.step(S[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                        alpha[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                        B[:,t_idx].unsqueeze(-1).repeat(1,Mtransitions),
                                        u_t[:,:,t_idx])
            timestep_tp1[:,:,t_idx] = t_idx+1

        # store (outer) trajectories in a dictionary
        trajs = {'S' : S, # starting and ending states -- asset price
                'alpha' : alpha, # starting and ending states -- amount of the stock held by the agent
                'B' : B, # starting and ending states -- bank account cash-flow
                'timestep' : timestep} # starting and ending states -- time index

        # store (inner) transitions in a dictionary
        transitions = {'S_tp1' : S_tp1, # ending states from the actions -- asset price
                        'alpha_tp1' : alpha_tp1, # ending states from the actions -- amount of the stock held by the agent
                        'B_tp1' : B_tp1, # ending states from the actions -- bank account cash-flow
                        'timestep_tp1' : timestep_tp1, # ending states from the actions -- time index
                        'cost_t' : cost_t, # costs from the actions
                        'u_t' : u_t, # actions taken
                        'log_prob_t' : log_prob_t} # log-prob from the actions

        return trajs, transitions

    # estimate the value function for all time steps (critic)
    def estimate_V(self,
                    Ntrajectories, # number of trajectories
                    Mtransitions, # number of transitions
                    batch_size=50, # batch size for the update
                    Nepochs=100, # number of epochs
                    rng_seed=None): # random seed
        # print progress
        log_message = '--Estimation of V--'
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message+'\n')
        batch_size = np.minimum(batch_size, Ntrajectories)

        # set V in training mode
        self.V.train()
        
        # generate full trajectories from policy
        trajs, transitions = self.sim_trajectories(Ntrajectories,
                                                    Mtransitions,
                                                    is_action_random = True,
                                                    seed=rng_seed)
        
        for epoch in range(Nepochs):
            # zero grad
            self.V.zero_grad()

            # sample a batch of states at time t+1
            batch_idx = np.random.choice(Ntrajectories, size=batch_size, replace=False)
            S_batch = trajs["S"][batch_idx, 1:]
            alpha_batch = trajs["alpha"][batch_idx, 1:]
            B_batch = trajs["B"][batch_idx, 1:]
            time_batch = trajs["timestep"][batch_idx, 1:]
            
            # compute predicted values
            obs_t = T.stack((S_batch,
                            alpha_batch,
                            time_batch),-1).detach()
            v_pred = self.V(obs_t).squeeze()
            
            # compute target values
            v_target = T.zeros(v_pred.shape, requires_grad=False)

            # value function at the next time step
            obs_tp1 = T.stack((transitions["S_tp1"][batch_idx, :, 1:-1],
                                transitions["alpha_tp1"][batch_idx, :, 1:-1],
                                transitions["timestep_tp1"][batch_idx, :, 1:-1]),-1)
            v_tp1 = self.V(obs_tp1.clone()).squeeze()
            cost_t = transitions["cost_t"][batch_idx, :, 1:-1]
            
            # value function for last time step
            v_target[:, -1] = self.env.settlement(S_batch[:,-1],
                                                    alpha_batch[:,-1],
                                                    B_batch[:,-1])
            
            # value function for other time steps
            v_target[:, :-1] = self.risk_measure.compute_risk(cost_t + v_tp1)
            
            # calculate the loss function
            v_target = v_target.detach().to(self.device)
            v_loss = self.V.loss(v_target, v_pred).to(self.device)
            v_loss.backward()
            self.V.optimizer.step()
            self.loss_history_V.append(v_loss.detach())
        
            # print progress

            if epoch % self.loss_print == 0 or epoch == Nepochs - 1:
                loss_history = self.loss_history_V[-self.loss_trail:]
                loss_history_cpu = [loss.cpu().detach().numpy() for loss in loss_history]  # Move tensor to CPU and detach from graph
                mean_loss = np.mean(loss_history_cpu)

                log_message= f" Epoch = {str(epoch)}, " \
                             f" Loss: {str(mean_loss)}"
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message+'\n')
        
        # set V in evaluation mode
        self.V.eval()


    # update the policy according to a batch of trajectories (actor)
    def update_policy(self,
                        Ntrajectories, # number of trajectories
                        Mtransitions, # number of transitions
                        batch_size=50, # batch size for the update
                        Nepochs=100, # number of epochs
                        rng_seed=None): # random seed
        # print progress
        log_message = '--Update of pi--'
        print(log_message)
        with open(self.LOG_FILE, 'a') as file:
            file.write(log_message+'\n')
        batch_size = np.minimum(batch_size, Ntrajectories)
        
        # set the policy in training mode
        self.policy.train()
        
        for epoch in range(Nepochs):
            # zero grad
            self.policy.zero_grad()
            
            # sample a batch of transitions
            trajs, transitions = self.sim_trajectories(batch_size,
                                                  Mtransitions,
                                                  is_action_random=True,
                                                  seed=rng_seed)

            # value function of the next time step
            obs_tp1 = T.stack((transitions["S_tp1"][:, :, :-1].clone(),
                                transitions["alpha_tp1"][:, :, :-1].clone(),
                                transitions["timestep_tp1"][:, :, :-1].clone()),-1)
            V_tp1 = self.V(obs_tp1.clone()).squeeze()
            
            # combine both the cost and value function
            V_loss = self.risk_measure.get_V_loss(transitions["cost_t"][:, :, :-1].detach()+V_tp1.detach(),
                                                    transitions["log_prob_t"][:, :, :-1])

            # loss for each initial state
            grad_loss = V_loss 
            
            # average over all initial states and times
            loss = T.mean(grad_loss)
            
            # optimization step
            loss.to(self.device).backward()
            self.policy.optimizer.step()

            # store the loss
            self.loss_history_policy.append(loss.detach())

            # print progress
            if epoch % self.loss_print == 0 or epoch == Nepochs - 1:
                loss_history = self.loss_history_policy[-self.loss_trail:]
                loss_history_cpu = [loss.cpu().detach().numpy() for loss in loss_history]  # Move tensor to CPU and detach from graph
                mean_loss = np.mean(loss_history_cpu)
                log_message = f'   Epoch =  {str(epoch)} , '\
                              f' Loss: {mean_loss}'
                print(log_message)
                with open(self.LOG_FILE, 'a') as file:
                    file.write(log_message+'\n')

        # set the policy in evaluation mode
        self.policy.eval()

    
    # plot the strategy at any point in the algorithm
    def plot_current_policy(self):
        fixed_alpha = 0.0
        fixed_B = self.env.params["B0"]

        hist2dim_pi = np.zeros([len(self.env.spaces["S_space"]), len(self.env.spaces["t_space"])-1])
        for S_idx, S_val in enumerate(self.env.spaces["S_space"]):
            for time_idx, time_val in enumerate(self.env.spaces["t_space"][:-1]):
                hist2dim_pi[len(self.env.spaces["S_space"])-S_idx-1, time_idx], _ = \
                        self.select_actions(T.Tensor([S_val]).to(self.device),
                                            T.tensor([fixed_alpha]).to(self.device),
                                            T.tensor([fixed_B]).to(self.device),
                                            T.tensor([time_idx]).to(self.device),
                                            'best')

        # plot the policy
        plt.imshow(hist2dim_pi,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["t_space"]), 
                        np.max(self.env.spaces["t_space"]),
                        np.min(self.env.spaces["S_space"]),
                        np.max(self.env.spaces["S_space"])],
                aspect='auto',
                vmin=-self.env.params["max_alpha"],
                vmax=self.env.params["max_alpha"])
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Best actions')
        plt.xlabel("Time")
        plt.ylabel("Price of the asset")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.GRAPH_FILE1, transparent=True)
        plt.clf()


    # plot the value function at any point in the algorithm
    def plot_current_V(self):
        fixed_alpha = 0.0

        hist2dim_V = np.zeros([len(self.env.spaces["S_space"]), len(self.env.spaces["t_space"])-1])
        for S_idx, S_val in enumerate(self.env.spaces["S_space"]):
            for time_idx, time_val in enumerate(self.env.spaces["t_space"][:-1]):
                obs = T.stack((S_val*T.ones(1),
                                fixed_alpha*T.ones(1),
                                time_val*T.ones(1)), -1)
                obs = obs.to(self.device)
                hist2dim_V[len(self.env.spaces["S_space"])-S_idx-1, time_idx] = self.V(obs)

        # plot the value function
        plt.imshow(hist2dim_V,
                interpolation='none',
                cmap=utils.cmap,
                extent=[np.min(self.env.spaces["t_space"]), 
                        np.max(self.env.spaces["t_space"]),
                        np.min(self.env.spaces["S_space"]),
                        np.max(self.env.spaces["S_space"])],
                aspect='auto')
        plt.rcParams.update({'font.size': 16})
        plt.rc('axes', labelsize=20)
        plt.title('Value function')
        plt.xlabel("Time")
        plt.ylabel("Price of the asset")
        plt.colorbar()
        plt.tight_layout()
        now = datetime.now()
        plt.savefig(self.GRAPH_FILE2, transparent=True)
        plt.clf()

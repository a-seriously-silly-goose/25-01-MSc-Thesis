"""
Actor Critic PPO implementation based on Ding et al. (2025)
Extends the base ActorCriticPG implementation with PPO-specific features
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from actor_critic_pg import ActorCriticPG


class ActorCriticPPO(ActorCriticPG):
    def __init__(
        self, repo, method, env, policy, V, risk_measure, hyperparameter_set, LOG_FILE
    ):
        super().__init__(
            repo, method, env, policy, V, risk_measure, hyperparameter_set, LOG_FILE
        )

        # PPO specific parameters
        self.clip_epsilon = 0.2
        self.n_actor_updates = 10
        self.n_critic_updates = 10
        self.gae_lambda = 0.95
        self.entropy_coef = 0.01

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        advantages = T.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update_policy(
        self, Ntrajectories, Mtransitions, batch_size=50, Nepochs=100, rng_seed=None
    ):
        """PPO policy update implementation"""
        batch_size = min(batch_size, Ntrajectories)

        for epoch in range(Nepochs):
            # Collect trajectories
            trajs, transitions = self.sim_trajectories(
                batch_size, Mtransitions, "random", rng_seed
            )

            # Get value estimates
            obs = T.stack(
                (
                    transitions["S_tp1"][:, :, :-1],
                    transitions["alpha_tp1"][:, :, :-1],
                    transitions["timestep_tp1"][:, :, :-1],
                ),
                -1,
            )

            values = self.V(obs).detach()
            next_values = T.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]

            # Compute advantages using GAE
            rewards = transitions["cost_t"][:, :, :-1]
            dones = T.zeros_like(
                rewards
            )  # In this case, episodes don't terminate early
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Store old policy distribution parameters
            old_actions_param1, old_actions_param2 = self.policy(obs)
            old_dist = Normal(old_actions_param1, old_actions_param2)
            old_log_probs = old_dist.log_prob(transitions["u_t"][:, :, :-1]).detach()

            # PPO updates
            for _ in range(self.n_actor_updates):
                self.policy.zero_grad()

                # Get current policy distribution
                actions_param1, actions_param2 = self.policy(obs)
                curr_dist = Normal(actions_param1, actions_param2)
                curr_log_probs = curr_dist.log_prob(transitions["u_t"][:, :, :-1])

                # Calculate ratios and PPO clip objective
                ratios = T.exp(curr_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = (
                    T.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )

                # Policy loss
                policy_loss = -T.min(surr1, surr2).mean()

                # Entropy loss for exploration
                entropy_loss = -self.entropy_coef * curr_dist.entropy().mean()

                # Total loss
                loss = policy_loss + entropy_loss

                loss.backward()
                self.policy.optimizer.step()

            # Value function updates
            for _ in range(self.n_critic_updates):
                self.V.zero_grad()
                value_pred = self.V(obs)
                value_loss = F.mse_loss(value_pred, returns.detach())
                value_loss.backward()
                self.V.optimizer.step()

            # Store losses
            self.loss_history_policy.append(loss.detach().cpu().numpy())

            # Print progress
            if epoch % self.loss_print == 0 or epoch == Nepochs - 1:
                mean_loss = np.round(
                    np.mean(self.loss_history_policy[-self.loss_trail :]), 4
                )
                log_message = f"   Epoch =  {str(epoch)} , Loss: {mean_loss}"
                print(log_message)
                with open(self.LOG_FILE, "a") as file:
                    file.write(log_message + "\n")

        # Set policy to evaluation mode
        self.policy.eval()

"""
Actor Critic PPO implementation based on Ding et al. (2025)
Extends the base ActorCriticPG implementation with PPO-specific features
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from agents.actor_critic_pg import ActorCriticPG


class ActorCriticPPO(ActorCriticPG):
    def __init__(
        self, repo, method, env, policy, V, risk_measure, hyperparameter_set, LOG_FILE
    ):
        super().__init__(
            repo, method, env, policy, V, risk_measure, hyperparameter_set, LOG_FILE
        )

        # PPO specific parameters
        self.clip_epsilon = 0.2
        self.n_actor_updates = 10  # n_actor_update in Algorithm 2
        self.n_critic_updates = 10  # n_critic_update in Algorithm 2
        self.gae_lambda = 0.95  # λ in GAE
        self.entropy_coef = 0.01  # λ_entropy in Algorithm 2

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (Algorithm 2 line 5)"""
        gae = 0
        advantages = T.zeros_like(rewards)

        for t in reversed(range(len(rewards))):
            # Check if next state is terminal
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = next_values[t]
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            # TD error (δ_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            # GAE: gae = δ_t + γλ * next_non_terminal * gae_{t+1}
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update_policy(
        self, Ntrajectories, Mtransitions, batch_size=50, Nepochs=100, rng_seed=None
    ):
        """PPO policy update (Algorithm 2)"""
        batch_size = min(batch_size, Ntrajectories)

        for epoch in range(Nepochs):
            # Collect trajectories with best current POLICY (Algorithm 2 line 2)
            trajs, transitions = self.sim_trajectories(
                batch_size,
                Mtransitions,
                "best",
                rng_seed,  # Fixed: "best" not "random"
            )

            # Construct state: [price, holdings, time] (ensure all components included)
            obs = T.cat(
                [
                    transitions["S_tp1"][:, :, :-1],
                    transitions["alpha_tp1"][:, :, :-1],
                    transitions["timestep_tp1"][:, :, :-1],
                ],
                dim=-1,
            )

            # Get value estimates V(s)
            values = self.V(obs).squeeze(-1).detach()  # Remove extra dim if needed

            # Prepare next values & terminal flags
            next_values = T.zeros_like(values)
            next_values[:, :-1] = values[:, 1:]
            dones = transitions["done_t"][:, :, :-1]  # Use actual terminal flags

            # Compute GAE advantages (Algorithm 2 line 5)
            rewards = -transitions["cost_t"][:, :, :-1]  # Negative cost = reward
            advantages, returns = self.compute_gae(rewards, values, next_values, dones)

            # Optional: Advantage normalization
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Store old log probabilities (θ_old in Algorithm 2 line 3)
            with T.no_grad():
                old_actions_param1, old_actions_param2 = self.policy(obs)
                old_dist = Normal(old_actions_param1, old_actions_param2)
                old_log_probs = old_dist.log_prob(transitions["u_t"][:, :, :-1])

            # PPO Actor Updates (Algorithm 2 lines 4-8)
            for _ in range(self.n_actor_updates):
                self.policy.zero_grad()

                # Current policy log probabilities
                actions_param1, actions_param2 = self.policy(obs)
                curr_dist = Normal(actions_param1, actions_param2)
                curr_log_probs = curr_dist.log_prob(transitions["u_t"][:, :, :-1])

                # Probability ratio (Algorithm 2 line 6)
                ratios = T.exp(curr_log_probs - old_log_probs.detach())

                # Clipped surrogate objective
                surr1 = ratios * advantages
                surr2 = (
                    T.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages
                )
                policy_loss = -T.min(surr1, surr2).mean()

                # Entropy loss (Algorithm 2 line 7)
                entropy_loss = -self.entropy_coef * curr_dist.entropy().mean()

                # Total loss and update (Algorithm 2 line 8)
                loss = policy_loss + entropy_loss
                loss.backward()
                self.policy.optimizer.step()

            # Critic Updates (Algorithm 2 lines 9-10)
            for _ in range(self.n_critic_updates):
                self.V.zero_grad()
                value_pred = self.V(obs).squeeze(-1)

                # Value loss = Σ(Φ²) (Algorithm 2 line 10), NOT returns!
                value_loss = (advantages.detach() ** 2).mean()  # Critical fix
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

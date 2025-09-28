import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import yaml
import os
from datetime import datetime


class ActorCriticPPO:
    def __init__(
        self,
        env,
        policy_net,
        value_net,
        hyperparameter_set="ppo_default",
        log_file="training.log",
    ):
        self.env = env
        self.policy = policy_net
        self.value_net = value_net
        self.log_file = log_file

        # Load hyperparameters
        with open(
            "/Users/simeon/Documents/GitHub/03 University/25 01 MSc Thesis/hyperparameters.yml",
            "r",
        ) as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        envParams = hyperparameters["envParams"]
        algoParams = hyperparameters["algoParams"]
        riskParams = hyperparameters["riskParams"]
        runParams = hyperparameters["runParams"]
        repo_name = hyperparameter_set

        self.gamma = algoParams["gamma"]
        self.clip_epsilon = algoParams["clip_epsilon"]
        self.gae_lambda = algoParams["gae_lambda"]
        self.entropy_coef = algoParams["entropy_coef"]
        self.n_epochs = algoParams["Nepochs"]
        self.batch_size = algoParams["batch_pi"]
        self.n_actor_updates = algoParams["n_actor_updates"]
        self.n_critic_updates = algoParams["n_critic_updates"]

        # Initialize optimizers
        self.policy_optim = T.optim.Adam(
            self.policy.parameters(), lr=algoParams["lr_V"]
        )
        self.value_optim = T.optim.Adam(
            self.value_net.parameters(), lr=algoParams["lr_pi"]
        )

        # Loss tracking
        self.policy_loss_history = []
        self.value_loss_history = []

    def log_message(self, message):
        print(message)
        with open(self.log_file, "a") as f:
            f.write(f"{datetime.now()} - {message}\n")

    def collect_trajectories(self, num_trajectories, max_steps=None):
        """
        Collect trajectories by rolling out the current policy.

        Args:
            num_trajectories: Number of trajectories to collect
            max_steps: Maximum number of steps per trajectory (defaults to env's Ndt if None)

        Returns:
            Dictionary containing collected trajectory data
        """
        if max_steps is None:
            max_steps = self.env.params["Ndt"]

        # Initialize containers for batch data
        all_states = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_values = []
        all_log_probs = []
        all_terminal_rewards = []

        for _ in range(num_trajectories):
            # Reset environment
            S_t, v_t, alpha_tm1, B_t = self.env.reset()

            states = []
            actions = []
            rewards = []
            dones = []
            values = []
            log_probs = []

            for t in range(max_steps):
                # Create state tensor with proper observations
                # Important: Use the same state representation as in behaviour_replication
                state = T.stack([S_t, alpha_tm1, T.tensor([t / max_steps])], dim=-1)

                # Get action from policy and its log probability
                with T.no_grad():
                    mu, sigma = self.policy(state)
                    # Sample action from Gaussian distribution
                    action_dist = T.distributions.Normal(mu, sigma)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                    # Get value estimate
                    value = self.value_net(state).squeeze()

                # Take step in environment
                S_tp1, v_tp1, alpha_t, B_tp1, reward = self.env.step(
                    S_t, v_t, alpha_tm1, B_t, action
                )

                # Store transition
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(t == max_steps - 1)  # Only true on last step
                values.append(value)
                log_probs.append(log_prob)

                # Update for next step
                S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

            # Calculate terminal reward
            terminal_reward = self.env.get_final_cost(S_t, v_t, alpha_tm1, B_t)
            all_terminal_rewards.append(terminal_reward)

            # Store trajectory
            all_states.append(T.stack(states))
            all_actions.append(T.stack(actions))
            all_rewards.append(T.stack(rewards))
            all_dones.append(T.tensor(dones))
            all_values.append(T.stack(values))
            all_log_probs.append(T.stack(log_probs))

        # Combine all trajectories
        batch_states = T.cat(all_states)
        batch_actions = T.cat(all_actions)
        batch_rewards = T.cat(all_rewards)
        batch_dones = T.cat(all_dones)
        batch_values = T.cat(all_values)
        batch_log_probs = T.cat(all_log_probs)

        return {
            "states": batch_states,
            "actions": batch_actions,
            "rewards": batch_rewards,
            "dones": batch_dones,
            "values": batch_values,
            "log_probs": batch_log_probs,
            "terminal_rewards": T.tensor(all_terminal_rewards),
        }

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (Algorithm 3b)"""
        advantages = T.zeros_like(rewards)
        gae = 0

        # Reverse through time steps
        for t in reversed(range(rewards.size(1))):
            if t == rewards.size(1) - 1:
                next_non_terminal = 1.0 - dones[:, t]
                next_value = next_values[:, t]
            else:
                next_non_terminal = 1.0 - dones[:, t]
                next_value = values[:, t + 1]

            # TD residual (Algorithm 3b step 2)
            delta = (
                rewards[:, t]
                + self.gamma * next_value * next_non_terminal
                - values[:, t]
            )

            # GAE accumulation (Algorithm 3b step 3)
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[:, t] = gae

        # Calculate returns (Algorithm 3 step 3)
        returns = advantages + values
        return advantages, returns

    def update(self, num_trajectories):
        """PPO update procedure (Algorithm 3)"""
        self.log_message("Starting PPO update")

        # Collect trajectories (Algorithm 3 step 1)
        data = self.collect_trajectories(num_trajectories)
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        old_log_probs = data["log_probs"]
        values = data["values"]
        next_values = data["next_values"]

        # Compute GAE and returns (Algorithm 3 step 2-3)
        advantages, returns = self.compute_gae(rewards, values, next_values, dones)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten batch and timesteps
        flat_states = states.view(-1, states.size(-1))
        flat_actions = actions.view(-1)
        flat_old_log_probs = old_log_probs.view(-1)
        flat_advantages = advantages.view(-1)
        flat_returns = returns.view(-1)

        # PPO updates
        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = T.randperm(flat_states.size(0))

            # Mini-batch updates
            for start in range(0, flat_states.size(0), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                # Get batch data
                batch_states = flat_states[batch_idx]
                batch_actions = flat_actions[batch_idx]
                batch_old_log_probs = flat_old_log_probs[batch_idx]
                batch_advantages = flat_advantages[batch_idx]
                batch_returns = flat_returns[batch_idx]

                # --- ACTOR UPDATE (Algorithm 3 steps 4-7) ---
                for _ in range(self.n_actor_updates):
                    # Get current policy distribution
                    action, new_log_probs = self.policy(batch_states)

                    # Probability ratio (Algorithm 3 step 4)
                    ratios = T.exp(new_log_probs - batch_old_log_probs)

                    # Clipped surrogate objective (Algorithm 3 step 4)
                    surr1 = ratios * batch_advantages
                    surr2 = (
                        T.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                        * batch_advantages
                    )
                    policy_loss = -T.min(surr1, surr2).mean()

                    # Entropy bonus (Algorithm 3 step 5) ##TODO: fix entropy
                    entropy_loss = (
                        0  # -self.entropy_coef * action_dist.entropy().mean()
                    )

                    # Total actor loss (Algorithm 3 step 6)
                    actor_loss = policy_loss + entropy_loss

                    # Update policy (Algorithm 3 step 7)
                    self.policy_optim.zero_grad()
                    actor_loss.backward()
                    self.policy_optim.step()

                # --- CRITIC UPDATE (Algorithm 4) ---
                for _ in range(self.n_critic_updates):
                    # Current value predictions
                    value_pred = self.value_net(batch_states).squeeze()

                    # Value loss (MSE against returns)
                    value_loss = F.mse_loss(value_pred, batch_returns)

                    # Update value network
                    self.value_optim.zero_grad()
                    value_loss.backward()
                    self.value_optim.step()

            # Store losses for logging
            self.policy_loss_history.append(actor_loss.item())
            self.value_loss_history.append(value_loss.item())

            # Log progress
            if epoch % 10 == 0:
                self.log_message(
                    f"Epoch {epoch+1}/{self.n_epochs} | "
                    f"Policy Loss: {actor_loss.item():.4f} | "
                    f"Value Loss: {value_loss.item():.4f}"
                )

        self.log_message("PPO update completed")
        return {
            "policy_loss": self.policy_loss_history,
            "value_loss": self.value_loss_history,
        }

    def save_models(self, save_dir):
        """Save policy and value networks"""
        os.makedirs(save_dir, exist_ok=True)
        T.save(self.policy.state_dict(), os.path.join(save_dir))
        T.save(self.value_net.state_dict(), os.path.join(save_dir))
        self.log_message(f"Models saved to {save_dir}")

    def load_models(self, load_dir):
        """Load policy and value networks"""
        self.policy.load_state_dict(T.load(os.path.join(load_dir)))
        # self.value_net.load_state_dict(T.load(os.path.join(load_dir)))
        self.log_message(f"Models loaded from {load_dir}")

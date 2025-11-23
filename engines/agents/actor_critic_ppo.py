import os
import torch as T
import torch.nn.functional as F
from datetime import datetime
from engines.agents.models import PolicyNN, CriticNN
from engines.environments import HestonEnv
from engines.gpu_manager import GPUMemoryManager


class ActorCriticPPO:
    def __init__(
        self,
        env : HestonEnv,
        policy_net: PolicyNN,
        value_net: CriticNN,
        algoParams: dict,
        hyperparameters_version: str,
        root_repo: str,
    ):
        self.gamma = algoParams.gamma
        self.clip_epsilon = algoParams.clip_epsilon
        self.gae_lambda = algoParams.gae_lambda
        self.entropy_coef = algoParams.entropy_coef
        self.n_epochs = algoParams.Nepochs
        self.batch_size = algoParams.batch_pi
        self.n_actor_updates = algoParams.n_actor_updates
        self.n_critic_updates = algoParams.n_critic_updates
        self.N_rollouts = algoParams.Ntrajectories * algoParams.Mtransitions
        self.K_crit = algoParams.n_critic_updates
        self.K_act = algoParams.n_actor_updates
        self.algo_params = algoParams

        self.env = env
        self.policy = policy_net
        self.value_net = value_net
        self.hyperparameters_version = hyperparameters_version
        self.loss_print = 100  # Print loss every n epochs

        # Determine device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value_net.to(self.device)

        # Initialize optimizers
        self.policy_optim = T.optim.Adam(
            self.policy.parameters(), lr=algoParams.lr_pi
        )
        self.value_optim = T.optim.Adam(
            self.value_net.parameters(), lr=algoParams.lr_V
        )

        # Repository path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.repo = os.path.join(root_repo, "runs", f"{self.hyperparameters_version}_BC_{timestamp}")

        # Paths for saving files
        self.LOG_FILE = os.path.join(self.repo, "log.txt")
        self.PI_MODEL_FILE = os.path.join(self.repo, "PI_model.pt")
        self.V_MODEL_FILE = os.path.join(self.repo, "V_model.pt")
        self.PLOT_DIR = os.path.join(self.repo, "plots")

        os.makedirs(self.repo, exist_ok=True)
        os.makedirs(self.PLOT_DIR, exist_ok=True)

        # GPU Memory Manager
        self.memory_manager = GPUMemoryManager()
        # self.batch_size = self.adaptive_batch_sizing()
        self.batch_size = 500  # Fixed batch size for now

        # Loss tracking
        self.policy_loss_history = []
        self.value_loss_history = []

    def log_message(self, message):
        print(message)
        with open(self.LOG_FILE, "a") as f:
            f.write(f"{datetime.now()} - {message}\n")

    # Helper: vectorized rollout for the *current* policy (policy_fn should return (mu, sigma))
    def _vectorized_rollout(self,batch_size, max_steps=None, device=None):
        """
        Runs `batch_size` trajectories in parallel using `policy_fn` (callable(state) -> (mu,sigma)).
        Returns dict with tensors shaped [T, B, ...] or [T, B] as appropriate.
        """
        if max_steps is None:
            max_steps = self.env.Ndt
        if device is None:
            device = next(self.policy.parameters()).device

        B = batch_size
        # Reset batched env
        S_t, alpha_tm1, B_t, t, v_t = self.env.reset()
        # ensure tensors on device
        S_t = S_t.to(device)
        v_t = v_t.to(device)
        alpha_tm1 = alpha_tm1.to(device)
        B_t = B_t.to(device)
        # Repeat to match batch size B if needed
        if S_t.shape[0] != B:
            S_t = S_t.repeat(B)
            v_t = v_t.repeat(B)
            alpha_tm1 = alpha_tm1.repeat(B)
            B_t = B_t.repeat(B)

        # Preallocate
        states = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        for t in range(max_steps):
            t_norm = T.full((B,), float(t) / float(max_steps), device=device)
            obs = T.stack([S_t, alpha_tm1, B_t, t_norm], dim=-1)  # shape [B, state_dim=3]

            with T.no_grad():
                mu, sigma = self.policy(obs)
                dist = T.distributions.Normal(mu, sigma)
                action = dist.rsample().squeeze(-1)  # rsample for reparam if needed; use sample() if not
                log_prob = dist.log_prob(action)[0]
                value = self.value_net(obs).squeeze(-1)

            # Step environment (vectorized)
            state_tp1, reward = self.env.step((S_t, alpha_tm1, B_t, t, v_t), action)
            S_tp1, alpha_t, B_tp1, t, v_tp1 = state_tp1

            # Move to device (if env returned cpu)
            S_tp1 = T.as_tensor(S_tp1, device=device)
            v_tp1 = T.as_tensor(v_tp1, device=device)
            alpha_t = T.as_tensor(alpha_t, device=device)
            B_tp1 = T.as_tensor(B_tp1, device=device)
            reward = T.as_tensor(reward, device=device)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            # done is only True on last time-step in your rollout scheme (or env can provide masks)
            dones.append(T.zeros(B, dtype=T.bool, device=device))
            values.append(value)
            log_probs.append(log_prob)

            # update
            S_t, v_t, alpha_tm1, B_t = S_tp1, v_tp1, alpha_t, B_tp1

        # terminal reward and final states
        terminal_reward = self.env.get_final_QHE(S_t, alpha_tm1, B_t)
        final_states = T.stack([S_t, alpha_tm1, B_t, T.full((B,), 1.0, device=device)], dim=-1)  # t_norm=1 for final
        terminal_reward = T.as_tensor(terminal_reward, device=device)

        # Stack: shape [T, B, ...]
        states = T.stack(states)           # [T, B, state_dim]
        actions = T.stack(actions)         # [T, B, action_dim?] or [T, B]
        rewards = T.stack(rewards)         # [T, B]
        dones = T.stack(dones)             # [T, B]
        values = T.stack(values)           # [T, B]
        log_probs = T.stack(log_probs)     # [T, B]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
            "final_states": final_states,          # [B, state_dim]
            "terminal_reward": terminal_reward,    # [B]
        }


    # GAE (expects inputs shaped [T, B])
    def compute_gae(self, rewards, values, next_value, dones):
        """
        Vectorized GAE: rewards [T, B], values [T, B], next_value [B], dones [T, B]
        Returns: advantages [T, B], returns [T, B]
        """
        T_steps, B = rewards.size()
        device = rewards.device
        advantages = T.zeros(T_steps, B, device=device)
        gae = T.zeros(B, device=device)

        for t in reversed(range(T_steps)):
            if t == T_steps - 1:
                nv = next_value
            else:
                nv = values[t + 1]
            next_non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.gamma * nv * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns


    # Algorithm 3: Update Critic (use frozen policy to generate D)
    def update_critic(self, n_rollouts, rollout_horizon=None, critic_epochs=None, batch_size=None):
        """
        Implements Algorithm 3 (Value Function Fine-Tuning).
        - n_rollouts: N in the algorithm (batch of parallel envs)
        - rollout_horizon: T
        - critic_epochs: K_crit,PPO
        - batch_size: minibatch for optimizer steps (optional)
        """
        device = next(self.value_net.parameters()).device
        rollout_horizon = rollout_horizon or self.env.params["Ndt"]
        critic_epochs = critic_epochs or self.K_crit  # use attribute K_crit
        batch_size = batch_size or self.batch_size

        # Freeze policy: collect rollouts with current policy parameters (no grads)
        self.policy.eval()
        with T.no_grad():
            # data = self.policy_rollout(num_trajectories=n_rollouts, track_policy_gradients=False)
            data = self._vectorized_rollout(batch_size=n_rollouts, max_steps=rollout_horizon, device=device)

        states = data["states"]       # [T, B, state_dim]
        rewards = data["rewards"]     # [T, B]
        dones = data["dones"]         # [T, B]
        # values as computed by current value_net BEFORE critic update (V_phi(s_t))
        values = data["values"]       # [T, B]
        final_states = data["final_states"]  # [B, state_dim]

        # Next value bootstrap using current value_net
        with T.no_grad():
            next_value = self.value_net(final_states).squeeze(-1)   # [B]

        # Compute GAE
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)

        # Optionally normalize advantages? Algorithm 3 doesn't require it, but it helps optimization.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Flatten for batching: [T*B, ...]
        T_steps, B = rewards.shape
        flat_states = states.reshape(-1, states.size(-1))
        flat_returns = returns.reshape(-1)
        flat_advantages = advantages.reshape(-1)

        num_samples = flat_states.size(0)

        # Training loop for critic
        value_loss_hist = []
        for epoch in range(critic_epochs):
            perm = T.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                idx = perm[start : start + batch_size]
                batch_states = flat_states[idx]
                batch_returns = flat_returns[idx]

                value_pred = self.value_net(batch_states).squeeze(-1)
                # Mean squared error to returns (equivalent to minimizing GAE^2)
                value_loss = F.mse_loss(value_pred, batch_returns)

                self.value_optim.zero_grad()
                value_loss.backward()
                self.value_optim.step()

            value_loss_hist.append(value_loss.item())

        # Return updated critic and diagnostics
        self.policy.train()
        return {"value_loss_hist": value_loss_hist}


    # Algorithm 4: Update Actor (use frozen critic to evaluate values)
    def update_actor(self, n_rollouts, rollout_horizon=None, actor_epochs=None, batch_size=None, n_actor_updates=1):
        """
        Implements Algorithm 4 (PPO Policy Fine-Tuning).
        - n_rollouts: Nrollouts used to generate trajectories for policy update
        - rollout_horizon: T
        - actor_epochs: K_act,PPO  (how many outer epochs to run)
        - n_actor_updates: number of internal PPO update passes on the collected data (per epoch)
        """
        device = next(self.policy.parameters()).device
        rollout_horizon = rollout_horizon or self.env.params["Ndt"]
        actor_epochs = actor_epochs or self.K_act  # attribute from config
        batch_size = batch_size or self.batch_size

        policy_loss_hist = []
        ent_hist = []

        # We'll freeze critic when computing value estimates
        self.value_net.eval()

        for epoch in range(actor_epochs):
            # Per Algorithm 4 you simulate rollouts of current policy for this epoch
            self.policy.eval()
            with T.no_grad():
                data = self._vectorized_rollout(batch_size=n_rollouts, max_steps=rollout_horizon, device=device)
                # data = self.policy_rollout(num_trajectories=n_rollouts, track_policy_gradients=False)
                states = data["states"]      # [T, B, state_dim]
                actions = data["actions"]    # [T, B]
                rewards = data["rewards"]    # [T, B]
                dones = data["dones"]        # [T, B]
                old_log_probs = data["log_probs"]  # [T, B]
                final_states = data["final_states"]  # [B, state_dim]

                if T.isnan(states).any() or T.isnan(actions).any() or T.isnan(rewards).any():
                    self.log_message("WARNING: NaN detected in rollout data!")
                    break

                # Critic (frozen) provides values used by Algorithm 4
                # Evaluate V~(s_t) for all states
                T_steps, B = states.shape[:2]
                flat_states = states.reshape(-1, states.size(-1))
                with T.no_grad():
                    flat_values = self.value_net(flat_states).squeeze(-1)
                values = flat_values.reshape(T_steps, B)

                # Next value for bootstrapping
                with T.no_grad():
                    next_value = self.value_net(final_states).squeeze(-1)

            # Compute GAE using frozen critic values
            advantages, returns = self.compute_gae(rewards, values, next_value, dones)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Flatten for updates
            flat_states = states.reshape(-1, states.size(-1))
            flat_actions = actions.reshape(-1, *actions.shape[2:]) if actions.dim() > 2 else actions.reshape(-1)
            flat_old_log_probs = old_log_probs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)

            num_samples = flat_states.size(0)
            perm = T.randperm(num_samples, device=device)

            # Perform PPO-style minibatch updates (you can set n_actor_updates=1 to match single-step Algorithm 4)
            for _update_pass in range(n_actor_updates):
                for start in range(0, num_samples, batch_size):
                    idx = perm[start : start + batch_size]
                    batch_states = flat_states[idx]
                    batch_actions = flat_actions[idx]
                    batch_old_log_probs = flat_old_log_probs[idx]
                    batch_advantages = flat_advantages[idx]

                    # Check for NaN in batch data
                    if T.isnan(batch_states).any() or T.isnan(batch_actions).any():
                        self.log_message("WARNING: NaN in batch data!")
                        continue

                    # Evaluate policy on batch
                    mu, sigma = self.policy(batch_states)
                                    # Add NaN check for policy outputs
                    if T.isnan(mu).any() or T.isnan(sigma).any():
                        self.log_message("CRITICAL: Policy outputting NaN values!")
                        self.log_message(f"mu stats: mean={mu.mean().item()}")
                        self.log_message(f"sigma stats: mean={sigma.mean().item()}")
                        for i in range(4):
                            self.log_message(f"batch_states[:5, {i}]: {batch_states[:5, i].cpu().numpy()}")
                    dist = T.distributions.Normal(mu, sigma)
                    new_log_probs = dist.log_prob(batch_actions)[0]
                    if new_log_probs.dim() > 1:
                        new_log_probs = new_log_probs.sum(-1)

                    # Ratio for PPO clip
                    ratios = T.exp(new_log_probs - batch_old_log_probs)

                    surr1 = ratios * batch_advantages
                    surr2 = T.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    policy_loss = -T.min(surr1, surr2).mean()

                    # Entropy bonus (Algorithm 4 Eq. 5.7)
                    entropy = dist.entropy()
                    if entropy.dim() > 1:
                        entropy = entropy.sum(-1)
                    entropy = entropy.mean()
                    entropy_loss = -self.entropy_coef * entropy

                    total_loss = policy_loss + entropy_loss

                    # Update actor
                    self.policy_optim.zero_grad()
                    total_loss.backward()
                    self.policy_optim.step()

                    policy_loss_hist.append(policy_loss.item())
                    ent_hist.append(entropy.item())

            self.policy.train()
            self.value_net.train()

        return {"policy_loss_hist": policy_loss_hist, "entropy_hist": ent_hist}


    # Top-level routine: orchestrates Algorithm 3 + Algorithm 4 as per your outer snippet
    def train_ppo(self, K_PPO):
        """
        Outer loop:
        for K_PPO epochs do:
        2.1 Clear grads and D = empty
        2.2 Freeze policy (π̃ = πθ)
        2.3 V_phi' <- Algorithm 3 (update critic using frozen policy)
        2.4 Freeze critic (V~ = V_phi')
        2.5 πθ' <- Algorithm 4 (update actor using frozen critic)
        """

        self.log_message(f"Starting critic pretraining for {self.algo_params.Nepochs_V_init} epochs...")

        for pre_epoch in range(self.algo_params.Nepochs_V_init):
            critic_info = self.update_critic(
                n_rollouts=self.N_rollouts,
                rollout_horizon=self.env.Ndt,
                critic_epochs=1,              # one epoch per loop
                batch_size=self.batch_size
            )

            if (pre_epoch + 1) % 100 == 0:
                last_loss = critic_info.get("value_loss_hist", [None])[-1]
                self.log_message(
                    f"[PPO][V init] Critic epoch {pre_epoch+1}/{self.algo_params.Nepochs_V_init} | "
                    f"Last critic loss: {last_loss}"
                )

        self.log_message("Starting PPO training...")

        # Histories
        all_policy_losses = []
        all_value_losses = []

        for epoch in range(K_PPO):
            # 2.2 Freeze policy: update critic using the frozen policy's rollouts
            critic_info = self.update_critic(n_rollouts=self.N_rollouts,
                                            rollout_horizon=self.env.Ndt,
                                            critic_epochs=self.K_crit,
                                            batch_size=self.batch_size)
            all_value_losses.extend(critic_info.get("value_loss_hist", []))

            # 2.4 Freeze critic (we simply won't update it during actor update)
            # 2.5 Update actor using Algorithm 4 with frozen critic
            actor_info = self.update_actor(n_rollouts=self.N_rollouts,
                                        rollout_horizon=self.env.Ndt,
                                        actor_epochs=self.K_act,
                                        batch_size=self.batch_size,
                                        n_actor_updates=self.n_actor_updates)
            all_policy_losses.extend(actor_info.get("policy_loss_hist", []))

            self.log_message(f"[PPO] PPO Epoch {epoch+1}/{K_PPO} | "
                                f"Critic losses (last): {critic_info.get('value_loss_hist', [])[-1] if critic_info.get('value_loss_hist') else None} | "
                                f"Actor loss (last): {actor_info.get('policy_loss_hist', [])[-1] if actor_info.get('policy_loss_hist') else None}")


            # Optional logging
            if (epoch + 1) % self.loss_print == 0:
                self.save_models(self.repo)
                self.log_message(f"[PPO] PPO Epoch {epoch+1}/{K_PPO} | "
                                f"Critic losses (last): {critic_info.get('value_loss_hist', [])[-1] if critic_info.get('value_loss_hist') else None} | "
                                f"Actor loss (last): {actor_info.get('policy_loss_hist', [])[-1] if actor_info.get('policy_loss_hist') else None}")

        return {"policy_loss": all_policy_losses, "value_loss": all_value_losses}


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

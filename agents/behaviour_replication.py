import torch as T
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.nn.functional import silu
from models import PolicyApprox


class BehaviorReplicationAgent:
    def __init__(
        self,
        repo,  # Repository for saving files
        method,  # Sub-directory name for the agent's outputs
        env,  # The environment object
        policy_structure,  # Policy Neural Network structure (like PG/PPO)
        V_structure,  # Value Function Neural Network structure
        hyperparameters_version,  # The version of the hyperparameters
        LOG_FILE,  # Log file for storing messages
    ):
        """
        Initialize the BehaviorReplicationAgent for hedging.

        This agent replicates the behavior of the Delta Hedge actor
        by replicating its outputs into the policy structure.
        """
        self.env = env
        self.policy = policy_structure
        self.V = V_structure  # Value function (may not be used directly)
        self.repo = repo
        self.method = method
        self.hyperparameters_version = hyperparameters_version
        self.device = self.policy.device
        self.LOG_FILE = LOG_FILE

        # Paths for saving files
        self.PI_MODEL_FILE = os.path.join(
            self.repo, f"Pi_{self.hyperparameters_version}.pt"
        )
        self.PLOT_DIR = os.path.join(self.repo, self.method, "plots")
        os.makedirs(self.PLOT_DIR, exist_ok=True)

    # Behavior cloning update with Gaussian policy
    def replicate_delta_hedging(
        self,
        delta_hedge_actor,
        epochs: int = 10000,
        lr: float = 1e-3,
        lambda_entropy: float = 0.01,
    ):
        """
        Pretrain by cloning the delta‐hedge actor with a Gaussian policy:
          a ~ N(μ_θ(s), σ_θ(s)^2).
        Loss = -E[log p(a_expert)]  + λ * (-Entropy)
        where p is Normal(μ,σ^2), and a_expert = Δ_market.
        """
        optimizer = T.optim.Adam(self.policy.parameters(), lr=lr)

        # build the grid once
        S_vals = T.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100).to(self.device)
        t_vals = T.linspace(0, self.env.T, 50).to(self.device)
        S_grid, t_grid = T.meshgrid(S_vals, t_vals, indexing="ij")
        v_fixed = T.full_like(S_grid, fill_value=self.env.sigma)
        alpha_fixed = T.zeros_like(S_grid)
        B_fixed = T.full_like(S_grid, fill_value=self.env.B0)

        num_points = S_grid.numel()

        for epoch in range(epochs):
            optimizer.zero_grad()
            ml_sum, ent_sum = 0.0, 0.0

            # loop over the grid
            for i in range(S_grid.shape[0]):
                for j in range(S_grid.shape[1]):
                    S_t = S_grid[i, j]
                    time_t = t_grid[i, j]

                    # 1) get expert Δ
                    with T.no_grad():
                        target_delta, _ = delta_hedge_actor.select_actions(
                            S_t,
                            v_fixed[i, j],
                            alpha_fixed[i, j],
                            B_fixed[i, j],
                            time_t,
                            choose="best",
                        )

                    # 2) run policy to get μ, σ
                    obs = T.stack([S_t, alpha_fixed[i, j], time_t]).unsqueeze(
                        0
                    )  # [1,3]
                    mu, sigma = self.policy(obs)  # each [1,1]
                    sigma = max(sigma, 1e-6)

                    # 3) log‐likelihood loss
                    dist = T.distributions.Normal(mu, sigma)
                    logp = dist.log_prob(target_delta)
                    ml_sum += -logp.sum()

                    # 4) entropy loss
                    ent = dist.entropy()
                    ent_sum += -ent.sum()

            # average over all grid points
            ml_loss = ml_sum / num_points
            ent_loss = ent_sum / num_points
            loss = ml_loss + lambda_entropy * ent_loss

            # backward & step
            loss.backward()
            optimizer.step()

            # logging + plotting every 10 epochs
            if epoch % 10 == 0 or epoch == epochs - 1:
                msg = f"[BC] Epoch {epoch}/{epochs} — ML={ml_loss.item():.5f}  Ent={ent_loss.item():.5f}"
                print(msg)
                with open(self.LOG_FILE, "a") as f:
                    f.write(msg + "\n")
                self.plot_current_policy(epoch)
                self.plot_action_vs_price()
                self.plot_action_vs_time()

    def replicate_delta_hedging2(self, delta_hedge_actor, epochs=1000):
        """
        Mimics the behavior of the Delta Hedge actor by training the policy
        to predict the same actions as the delta hedge actor for various states.

        delta_hedge_actor: The actor to mimic.
        epochs: Number of epochs for replication training.
        """
        optimizer = T.optim.Adam(self.policy.parameters(), lr=1e-3)
        loss_func = T.nn.MSELoss()

        # Generate grid for training
        S_vals = T.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100).to(self.device)
        t_vals = T.linspace(0, self.env.T, 50).to(self.device)
        S_grid, t_grid = T.meshgrid(S_vals, t_vals, indexing="ij")
        v_fixed = T.tensor(self.env.sigma, dtype=T.float32, device=self.device)
        alpha_fixed = T.tensor(0.0, dtype=T.float32, device=self.device)
        B_fixed = T.tensor(self.env.B0, dtype=T.float32, device=self.device)

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0

            for i in range(S_grid.shape[0]):
                for j in range(S_grid.shape[1]):
                    S_t = S_grid[i, j]
                    time_t = t_grid[i, j]

                    # Get delta hedge actor's action
                    with T.no_grad():
                        target_delta, _ = delta_hedge_actor.select_actions(
                            S_t, v_fixed, alpha_fixed, B_fixed, time_t, choose="best"
                        )

                    # Run through the current policy network
                    obs_t = T.tensor(
                        [S_t, alpha_fixed, time_t], device=self.device
                    ).unsqueeze(0)
                    predicted_action, _ = self.policy(obs_t)

                    # Compute the loss
                    total_loss += loss_func(predicted_action, target_delta)

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Log progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                log_message = f"Epoch {epoch}/{epochs}, Loss = {total_loss.item():.6f}"
                print(log_message)
                with open(self.LOG_FILE, "a") as log_file:
                    log_file.write(log_message + "\n")
                self.plot_current_policy(epoch)
                self.plot_action_vs_price()
                self.plot_action_vs_time()

    def plot_current_policy(self, epoch):
        """
        Plot the current policy learned by the agent.
        """
        S_vals = np.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100)
        t_vals = np.linspace(0, self.env.T, 50)
        S_grid, t_grid = np.meshgrid(S_vals, t_vals)
        v_fixed = self.env.sigma
        alpha_fixed = 0.0
        B_fixed = self.env.B0

        deltas = np.zeros_like(S_grid)
        for i in range(S_grid.shape[0]):
            for j in range(S_grid.shape[1]):
                S_t = T.tensor(S_grid[i, j], dtype=T.float32, device=self.device)
                time_t = T.tensor(t_grid[i, j], dtype=T.float32, device=self.device)
                obs_t = T.stack((S_t, T.tensor(alpha_fixed), time_t), dim=-1).unsqueeze(
                    0
                )

                with T.no_grad():
                    predicted_action, _ = self.policy(obs_t)

                deltas[i, j] = predicted_action.cpu().item()

        plt.figure(figsize=(10, 6))
        contour = plt.contourf(S_grid, t_grid, deltas, cmap="viridis")
        plt.colorbar(contour, label="Replicated Delta")
        plt.xlabel("Stock Price")
        plt.ylabel("Time to Maturity")
        plt.title(f"Replicated Delta: Epoch {epoch}")
        plot_file = os.path.join(self.PLOT_DIR, f"current_policy.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Policy plot for epoch {epoch} saved to {plot_file}")

    def plot_action_vs_price(self, time_fixed=None):
        """
        Plot the action (output of the policy) against stock price.
        This assumes a fixed time for the plot.

        Parameters:
        - time_fixed: The fixed time for plotting, defaults to the mid-point of `env.T` if None.
        """
        if time_fixed is None:
            time_fixed = self.env.T / 2  # Use the mid-point of the time horizon

        # Convert the fixed values to tensors with consistent shape
        time_fixed = T.tensor([time_fixed], dtype=T.float32, device=self.device)
        alpha_fixed = T.tensor([0.0], dtype=T.float32, device=self.device)
        B_fixed = T.tensor([self.env.B0], dtype=T.float32, device=self.device)

        # Grid for Stock Prices
        S_values = np.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100)
        actions = []

        for S in S_values:
            S_t = T.tensor(
                [S], dtype=T.float32, device=self.device
            )  # Ensure S_t has shape [1]
            obs_t = T.stack(
                (S_t, alpha_fixed, time_fixed), dim=-1
            )  # Now all tensors match in shape

            with T.no_grad():
                action, _ = self.policy(obs_t)
                actions.append(action.item())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(S_values, actions, label=f"Time (t) = {time_fixed.item():.2f}")
        plt.xlabel("Stock Price")
        plt.ylabel("Action (Hedge Position)")
        plt.title("Action vs Stock Price")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file = os.path.join(self.repo, self.method, "action_vs_price.png")
        plt.savefig(plot_file)
        print(f"Action vs Price plot saved at: {plot_file}")
        plt.close()

    def plot_action_vs_time(self, stock_price_fixed=None):
        """
        Plot the action (output of the policy) against time.
        This assumes a fixed stock price for the plot.

        Parameters:
        - stock_price_fixed: The fixed stock price for plotting, defaults to `env.S0` if None.
        """
        if stock_price_fixed is None:
            stock_price_fixed = self.env.S0  # Use the initial stock price

        # Convert fixed values to tensors with consistent shapes
        stock_price_fixed = T.tensor(
            [stock_price_fixed], dtype=T.float32, device=self.device
        )
        alpha_fixed = T.tensor([0.0], dtype=T.float32, device=self.device)
        B_fixed = T.tensor([self.env.B0], dtype=T.float32, device=self.device)

        # Time values for plotting
        time_values = np.linspace(0, self.env.T, 50)
        actions = []

        for time in time_values:
            time_t = T.tensor(
                [time], dtype=T.float32, device=self.device
            )  # Ensure consistent shape
            obs_t = T.stack(
                (stock_price_fixed, alpha_fixed, time_t), dim=-1
            )  # All tensors have shape [1]

            with T.no_grad():
                action, _ = self.policy(obs_t)
                actions.append(action.item())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(
            time_values,
            actions,
            label=f"Stock Price (S) = {stock_price_fixed.item():.2f}",
        )
        plt.xlabel("Time")
        plt.ylabel("Action (Hedge Position)")
        plt.title("Action vs Time")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file = os.path.join(self.repo, self.method, "action_vs_time.png")
        plt.savefig(plot_file)
        print(f"Action vs Time plot saved at: {plot_file}")
        plt.close()

    def save_policy(self):
        """
        Saves the current policy model to a file.
        """
        T.save(self.policy.state_dict(), self.PI_MODEL_FILE)
        print(f"Policy model saved at: {self.PI_MODEL_FILE}")

    def load_policy(self, model_file):
        """
        Loads a policy model from a file.
        """
        self.policy.load_state_dict(T.load(model_file))
        print(f"Policy model loaded from: {model_file}")

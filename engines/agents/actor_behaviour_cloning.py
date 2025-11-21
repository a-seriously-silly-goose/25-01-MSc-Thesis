import torch as T
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.nn.functional import silu
from engines.agents.models import PolicyNN
from engines.environments import BlackScholesEnv, HestonEnv
from scipy.stats import norm
import numpy as np
from pathlib import Path


class BehaviorReplicationAgent:
    def __init__(
        self,
        env: BlackScholesEnv,  # The environment object
        policy_net: PolicyNN ,  # Policy Neural Network structure (like PG/PPO)
        hyperparameters_version: str,  # The version of the hyperparameters
        root_repo: str ,  # Root repository path
    ):
        """
        Initialize the BehaviorReplicationAgent for hedging.

        This agent replicates the behavior of the Delta Hedge actor
        by replicating its outputs into the policy structure.
        """
        self.env = env
        self.policy = policy_net
        self.hyperparameters_version = hyperparameters_version
        self.loss_print = 100  # Print loss every n epochs

        # Repository path
        self.repo = root_repo

        # Determine device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # Paths for saving files
        self.LOG_FILE = os.path.join(
            self.repo, f"BC_log_{self.hyperparameters_version}.txt"
        )

        self.PI_MODEL_FILE = os.path.join(
            self.repo, f"PI_BC_{self.hyperparameters_version}.pt"
        )

        self.PLOT_DIR = os.path.join(self.repo, "BC", "plots")
        os.makedirs(self.PLOT_DIR, exist_ok=True)

    def simulate_delta_hedge_trajectory(self, batch_size: int):
        """
        Simulate a batch of trajectories using the Delta Hedge actor.

        Parameters:
        - batch_size: Number of trajectories to simulate.

        Returns:
        - batch: A tensor of trajectories with shape 
                 (batch_size, trajectory_length, state_dim + 1), where the last dimension
                 includes (S_t, v_t, alpha_t, B_t, time_t, delta_t).
        """
        # Create meshgrid for stock prices and time
        S_vals = T.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, batch_size, device=self.device)
        t_vals = T.linspace(0, self.env.T, self.env.Ndt, device=self.device)
        S_grid, t_grid = T.meshgrid(S_vals, t_vals, indexing="ij")

        # Randomly sample alpha_t and B_t
        alpha_t = T.rand(S_grid.shape, device=self.device)
        B_t = T.rand(S_grid.shape, device=self.device)

        # Volatility is constant in this environment
        v_t = T.full(S_grid.shape, self.env.sigma, device=self.device)

        # Stack the state components
        state = T.stack([S_grid, alpha_t, B_t, t_grid,  v_t], dim=-1)

        # Compute delta hedge for all states
        delta_t = self.delta_hedge(state.view(-1, state.shape[-1])).view(state.shape[:-1])

        # Add delta_t to the state tensor
        batch = T.cat([state[..., :5], delta_t.unsqueeze(-1)], dim=-1)

        return batch

    def delta_hedge(self, state):
        """
        Compute the Delta Hedge action for the given state.

        Parameters:
        - S_t: Stock price at time t.
        - v_t: Volatility at time t.
        - alpha_t: Current hedge position.
        - B_t: Current cash balance.
        - time_t: Current time.

        Returns:
        - delta: The hedge position suggested by the Delta Hedge strategy.
        """
        S_t, alpha_t, B_t, time_t, v_t = state[..., 0], state[..., 1], state[..., 2], state[..., 3], state[..., 4]

        K = self.env.K
        T_min_t = self.env.T - time_t
        r = self.env.r

        d1 = (T.log(S_t / K) + (r + 0.5 * v_t ** 2) * T_min_t) / (v_t * T.sqrt(T_min_t))
        delta = norm.cdf(d1)

        return T.as_tensor(delta, dtype=T.float32, device=self.device)

    # Behavior cloning update with Gaussian policy
    def train_behavior_cloning(
        self,
        epochs: int,
        lr: float,
        lambda_entropy: float,
        batch_size: int ,
    ):
        """
        This follows Algo 2 from the paper
        """
        optimizer = T.optim.Adam(self.policy.parameters(), lr=lr)
        loss_history = []

        self.policy.train()

        msg = f"Starting Behavior Cloning training for {epochs} epochs..."
        print(msg)
        with open(self.LOG_FILE, "a") as f:
            f.write(msg + "\n")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # 2. simulate full expert trajectory
            with T.no_grad():
                expert_trajectory = self.simulate_delta_hedge_trajectory(
                    batch_size=batch_size
                )  # shape: [1, T, state_dim + 1]

            # 3. get actor outcome on the trajectory
            mu, sigma = self.policy.forward(
                expert_trajectory[..., :4].squeeze(0))  # shape: [T, 1]

            # 4. compute MLE losses 
            loss = self.get_total_loss(mu, sigma, expert_trajectory[..., 5].squeeze(0), lambda_entropy)
            loss_history.append(loss.detach().numpy())


            # 6. backward & step
            loss.backward()
            optimizer.step()

            optimizer.step()

            # logging + plotting every 10 epochs
            if epoch % self.loss_print == 0 or epoch == epochs - 1:
                msg = f"[BC] Epoch {epoch}/{epochs} — Loss={loss.item():.5f}"
                print(msg)
                with open(self.LOG_FILE, "a") as f:
                    f.write(msg + "\n")
                self.plot_current_policy(epoch)
                self.plot_action_vs_price()
                self.plot_action_vs_time()
                self.save_policy()

            self.policy.eval()

    def get_total_loss(self, mu, sigma, expert_actions, lambda_entropy):
        """
        Compute the total loss for behavior cloning.

        Parameters:
        - mu: Mean actions predicted by the policy.
        - sigma: Standard deviation of actions predicted by the policy.
        - expert_actions: Actions taken by the expert (Delta Hedge).

        Returns:
        - total_loss: The combined loss from MLE and entropy.
        """
        # Maximum Likelihood Estimation (MLE) Loss
        expert_actions = expert_actions.view_as(mu)  # Ensure dimensions match
        MLE_loss = 0.5 * T.mean(((expert_actions - mu) / sigma) ** 2 + 2 * T.log(sigma) + T.log(T.tensor(2) * T.pi))
        
        Entropy_loss = 0.5 *T.mean(T.log(sigma**2 * 2 * T.pi * T.e))

        total_loss = MLE_loss - lambda_entropy * Entropy_loss  # Combine losses with entropy regularization
        return total_loss


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
                obs_t = T.stack((S_t, T.tensor(alpha_fixed), T.zeros_like(S_t), time_t), dim=-1).unsqueeze(
                    0
                )

                with T.no_grad():
                    predicted_action, _ = self.policy.forward(obs_t)

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

    
    def plot_action_vs_price(self, time_fixed=None):
        """
        Plot the action (output of the policy) against stock price.
        This assumes a fixed time for the plot.

        Parameters:
        - time_fixed: The fixed time for plotting, defaults to the mid-point of `env.T` if None.
        """
        if time_fixed is None:
            time_fixed = self.env.T / 2  # Use the mid-point of the time horizon

        # Grid for Stock Prices
        S_values = np.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, 100)
        actions = []

        for S in S_values:
            S_t = T.tensor(S, dtype=T.float32, device=self.device)
            time_t = T.tensor(time_fixed, dtype=T.float32, device=self.device)
            obs_t = T.stack((S_t, T.tensor(0.0), T.tensor(0.0), time_t), dim=-1).unsqueeze(0)

            with T.no_grad():
                action, _ = self.policy(obs_t)
                actions.append(action.item())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(S_values, actions, label=f"Time (t) = {time_fixed:.2f}")
        plt.xlabel("Stock Price")
        plt.ylabel("Action (Hedge Position)")
        plt.title("Action vs Stock Price")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file = os.path.join(self.PLOT_DIR, "action_vs_price.png")
        plt.savefig(plot_file)
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

        # Time values for plotting
        time_values = np.linspace(0, self.env.T, 50)
        actions = []

        for time in time_values:
            S_t = T.tensor(stock_price_fixed, dtype=T.float32, device=self.device)
            time_t = T.tensor(time, dtype=T.float32, device=self.device)
            obs_t = T.stack((S_t, T.tensor(0.0), T.tensor(0.0), time_t), dim=-1).unsqueeze(0)

            with T.no_grad():
                action, _ = self.policy(obs_t)
                actions.append(action.item())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(
            time_values,
            actions,
            label=f"Stock Price (S) = {stock_price_fixed:.2f}",
        )
        plt.xlabel("Time")
        plt.ylabel("Action (Hedge Position)")
        plt.title("Action vs Time")
        plt.legend()
        plt.grid(True)

        # Save the plot
        plot_file = os.path.join(self.PLOT_DIR, "action_vs_time.png")
        plt.savefig(plot_file)
        plt.close()

    def save_policy(self):
        """
        Saves the current policy model to a file.
        """
        T.save(self.policy.state_dict(), self.PI_MODEL_FILE)

    def load_policy(self, model_file):
        """
        Loads a policy model from a file.
        """
        self.policy.load_state_dict(T.load(model_file))
        print(f"Policy model loaded from: {model_file}")

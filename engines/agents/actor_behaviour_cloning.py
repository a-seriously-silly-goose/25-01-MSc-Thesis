import torch as T
import numpy as np
import os
import matplotlib.pyplot as plt
from engines.agents.models import PolicyNN
from engines.environments import BlackScholesEnv
from engines.gpu_manager import GPUMemoryManager
import math
from datetime import datetime


class BehaviorReplicationAgent:
    def __init__(
        self,
        env: BlackScholesEnv,  # The environment object
        policy_net: PolicyNN ,  # Policy Neural Network structure (like PG/PPO)
        algoParams: dict,  # Algorithm parameters
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

        # Determine device
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Load hyperparameters
        self.policy_optim = T.optim.Adam(self.policy.parameters(), lr=algoParams.lr_pi)

        # Repository path
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        self.repo = os.path.join(root_repo, "runs", f"{self.hyperparameters_version}_BC_{timestamp}")

        # Paths for saving files
        self.LOG_FILE = os.path.join(self.repo, "log.txt")
        self.PI_MODEL_FILE = os.path.join(self.repo, "BC_PI_model.pt")
        self.PLOT_DIR = os.path.join(self.repo, "plots")

        os.makedirs(self.repo, exist_ok=True)
        os.makedirs(self.PLOT_DIR, exist_ok=True)

        # GPU Memory Manager
        self.memory_manager = GPUMemoryManager()
        self.batch_size = self.adaptive_batch_sizing()
        self.batch_size = 500  # Fixed batch size for now

        
    def measure_memory_per_sample(self):
        """Measure actual memory usage per sample"""
        if not T.cuda.is_available():
            return 1024 * 1024  # Default 1MB if no GPU
        
        # Create dummy input with your typical state dimensions
        # Based on your code: [S_t, alpha_t, B_t, time_t] = 5 state variables
        dummy_state = T.randn(1, 4, device=self.device, dtype=T.float32)  # 1 sample
        
        # Measure memory before forward pass
        T.cuda.empty_cache()
        mem_before = T.cuda.memory_allocated()
        
        # Forward pass (this will allocate memory for activations)
        with T.amp.autocast(device_type=self.device.type):  # If using mixed precision
            mu, sigma = self.policy(dummy_state)
        
        # Optional: backward pass to measure training memory
        loss = mu.sum() + sigma.sum()  # Dummy loss
        loss.backward()
        
        mem_after = T.cuda.memory_allocated()
        
        # Memory per sample = total allocated / number of samples
        memory_per_sample = (mem_after - mem_before)
        
        print(f"Measured memory per sample: {memory_per_sample / 1024:.2f} KB")
        
        # Clear gradients and cache
        self.policy.zero_grad()
        T.cuda.empty_cache()
        
        return memory_per_sample

    def adaptive_batch_sizing(self):
        """Dynamically adjust batch size based on available memory"""
        if T.cuda.is_available():
            total_memory = T.cuda.get_device_properties(0).total_memory
            available_memory = total_memory - T.cuda.memory_allocated()
            
            # Measure memory per sample
            memory_per_sample = self.measure_memory_per_sample()
            
            # Add 20% buffer for overhead
            memory_per_sample_with_buffer = memory_per_sample * 1.2
            
            max_batch_size = int(available_memory * 0.7 / memory_per_sample_with_buffer)
            
            print(f"Adaptive Batch Sizing: Available {available_memory/1024**3:.2f}GB, "
                f"Per sample: {memory_per_sample/1024**2:.2f}MB, "
                f"Batch size: {max_batch_size}")
            
            return int(max(1, min(max_batch_size, 1024)))
        else:
            return 32
    
    def simulate_delta_hedge_trajectory(self, batch_size: int):
        """Vectorized delta hedge simulation"""
        # Use larger but fewer batches to reduce overhead
        S_vals = T.linspace(0.5 * self.env.S0, 1.5 * self.env.S0, batch_size, 
                           device=self.device, dtype=T.float32)
        t_vals = T.linspace(0, self.env.T, self.env.Ndt, 
                           device=self.device, dtype=T.float32)
        
        S_grid, t_grid = T.meshgrid(S_vals, t_vals, indexing="ij")
        
        # Use float32 to save memory
        alpha_t = T.rand(S_grid.shape, device=self.device, dtype=T.float32)
        B_t = T.rand(S_grid.shape, device=self.device, dtype=T.float32)
        v_t = T.full(S_grid.shape, self.env.sigma, device=self.device, dtype=T.float32)
        
        # Flatten for batch processing
        flat_states = T.stack([
            S_grid.flatten(), 
            alpha_t.flatten(), 
            B_t.flatten(), 
            t_grid.flatten(), 
            v_t.flatten()
        ], dim=-1)
        
        # Vectorized delta computation
        with T.no_grad():
            deltas = self.delta_hedge(flat_states)
        
        # Reshape back
        batch = T.cat([
            flat_states.reshape(*S_grid.shape, -1),
            deltas.reshape(*S_grid.shape, 1)
        ], dim=-1)
        
        return batch
    
    def normal_cdf(self,x):
        return 0.5 * (1.0 + T.erf(x / math.sqrt(2)))


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
        delta = self.normal_cdf(x=d1) 
        return delta

    # Behavior cloning update with Gaussian policy
    def train_behavior_cloning(
        self,
        epochs: int,
        lambda_entropy: float,
        batch_size: int,
    ):
        loss_history = []

        self.policy.train()

        print(f"Starting Behavior Cloning training for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.policy_optim.zero_grad()

            # Simulate expert trajectory
            with T.no_grad():
                expert_trajectory = self.simulate_delta_hedge_trajectory(batch_size=batch_size)

            # Extract states and actions - CORRECTED dimensions
            states = expert_trajectory[..., :4]  # All 5 state variables
            expert_actions = expert_trajectory[..., 5]  # Delta at index 5
            
            # Get batch dimensions
            batch_size_dim, time_steps, state_dim = states.shape
            
            # Reshape for batch processing
            states_flat = states.reshape(-1, state_dim)
            expert_actions_flat = expert_actions.reshape(-1)
            
            # Forward pass
            mu, sigma = self.policy(states_flat)
            
            # Ensure proper shapes
            mu = mu.squeeze(-1)  # [batch*time, 1] -> [batch*time]
            sigma = sigma.squeeze(-1)  # [batch*time, 1] -> [batch*time]
            
            # Compute loss
            loss = self.get_total_loss(mu, sigma, expert_actions_flat, lambda_entropy)
            loss_history.append(loss.detach().cpu().numpy())

            # Backward pass
            loss.backward()
            self.policy_optim.step()

            if epoch % (self.loss_print // 2) == 0:
                mem_info = self.memory_manager.get_memory_info()
                if mem_info:
                    print(f"[BC] Epoch {epoch}: {mem_info}")
                self.memory_manager.clear_cache()

            # Logging
            if epoch % self.loss_print == 0 or epoch == epochs - 1:
                print(f"[BC] Epoch {epoch}/{epochs} | Loss={loss.item():.5f}")
                with open(self.LOG_FILE, "a") as log_f:
                    log_f.write(f"{epoch},{loss.item():.5f}\n")

                self.plot_current_policy(epoch)
                self.plot_action_vs_price()
                self.plot_action_vs_time()
                self.plot_loss_history(loss_history)
                self.save_policy()

        self.policy.eval()
        with open(self.LOG_FILE, "a") as log_f:
            log_f.write(f"Training completed at {datetime.now()}\n")

        return loss_history

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

    def plot_loss_history(self, loss_history):
        """
        Plot the loss history over training epochs.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, label="Behavior Cloning Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Behavior Cloning Loss History")
        plt.legend()
        plt.grid(True)
        plot_file = os.path.join(self.PLOT_DIR, "BC_loss_history.png")
        plt.savefig(plot_file)
        plt.close()

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
                obs_t = T.stack((S_t, T.tensor(alpha_fixed, device=self.device), T.zeros_like(S_t), time_t), dim=-1).unsqueeze(
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
            obs_t = T.stack((S_t, T.tensor(0.0, device=self.device), T.tensor(0.0, device=self.device), time_t), dim=-1).unsqueeze(0)

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
            obs_t = T.stack((S_t, T.tensor(0.0, device=self.device), T.tensor(0.0, device=self.device), time_t), dim=-1).unsqueeze(0)

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

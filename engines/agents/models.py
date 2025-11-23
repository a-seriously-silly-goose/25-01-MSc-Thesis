"""
Models -- Neural Networks
Policy and value function with fully-connected ANNs

"""

# numpy

# pytorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dto.input_dtos import AlgoParams
from engines.environments import BaseEnv
def normalize_features(x, env):
    # Normalize features with environment parameters
    x_normalized = x.clone()
    x_normalized[..., 0] = (x[..., 0] - env.S0) / env.S0  # Price: normalize around S0
    x_normalized[..., 1] = x[..., 1] / env.max_alpha  # Hedge position: normalize to [-1, 1]
    x_normalized[..., 2] = x[..., 2] / (env.S0 * env.max_alpha)  # Bank account: scale appropriately
    x_normalized[..., 3] = x[..., 3]  # Time is already normalized
    return x_normalized

class PolicyNN(nn.Module):
    def __init__(self, input_dto: AlgoParams, env: BaseEnv):
        super(PolicyNN, self).__init__()
        self.env = env
        self.input_size = 4
        self.output_size = 1
        self.n_layers = input_dto.layers_pi
        self.hidden_size = input_dto.hidden_pi
        self.learn_rate = input_dto.lr_pi

        # Build layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for i in range(self.n_layers - 1)
        ])
        # Separate heads for mean and standard deviation
        self.mu_head = nn.Linear(self.hidden_size, self.output_size)
        self.sigma_head = nn.Linear(self.hidden_size, self.output_size)

        # Better weight initialization
        self._initialize_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def _initialize_weights(self):
        # Use Xavier/Glorot initialization for better gradient flow
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0.0)
        
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        # Initialize output heads
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.01)  # Small initial outputs
        nn.init.constant_(self.mu_head.bias, 0.0)
        
        nn.init.xavier_uniform_(self.sigma_head.weight, gain=0.01)
        nn.init.constant_(self.sigma_head.bias, 0.0)  # Start with small positive values

    def forward(self, x):
        # Normalize input features
        x = normalize_features(x, self.env)
        x = x.squeeze(-1) if x.dim() > 2 else x

        # Forward pass
        h = F.silu(self.layer1(x))
        for layer in self.hidden_layers:
            h = F.silu(layer(h))

        # Mean output (clamped to action space)
        mu = self.mu_head(h)
        mu = T.clamp(mu, -self.env.max_alpha, self.env.max_alpha)

        # Learnable standard deviation (ensure it's positive)
        sigma = F.softplus(self.sigma_head(h)) + 1e-6  # Always positive
        sigma = T.clamp(sigma, 1e-4, 1.0)  # Prevent too small/large std dev

        return mu, sigma

class CriticNN(nn.Module):
    def __init__(self, input_dto: AlgoParams, env: BaseEnv):
        super(CriticNN, self).__init__()
        self.env = env
        self.input_size = 4
        self.output_size = 1
        self.n_layers = input_dto.layers_V
        self.hidden_size = input_dto.hidden_V
        self.learn_rate = input_dto.lr_V

        # Build layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(self.n_layers - 1)
        ])
        self.layerN = nn.Linear(self.hidden_size, self.output_size)

        # Better weight initialization
        self._initialize_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.loss = nn.MSELoss()
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0.0)
        
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.xavier_uniform_(self.layerN.weight)
        nn.init.constant_(self.layerN.bias, 0.0)

    def forward(self, x):
        # Normalize input features (same as policy)
        x = normalize_features(x, self.env)
        x = x.squeeze(-1) if x.dim() > 2 else x

        h = F.silu(self.layer1(x))
        for layer in self.hidden_layers:
            h = F.silu(layer(h))

        value = self.layerN(h)
        return value
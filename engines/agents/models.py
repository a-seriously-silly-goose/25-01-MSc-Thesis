"""
Models -- Neural Networks
Policy and value function with fully-connected ANNs

"""

# numpy
import numpy as np

# pytorch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dto.input_dtos import AlgoParams
from engines.environments import BaseEnv

# normalize features of the neural nets
def normalize_features(x, env):
    # normalize features with environment parameters
    x[..., 0] = 2 * x[..., 0] / env.S0 - 1.0  # price
    x[..., 1] /= env.max_alpha  # actual hedge position
    x[..., 2] /= env.Ndt  # time

    return x


# build a fully-connected neural net for the policy
class PolicyNN(nn.Module):
    # constructor
    def __init__(self,  input_dto: AlgoParams, env: BaseEnv):
        super(PolicyNN, self).__init__()
        # input arguments
        self.env = env
        self.input_size = 4 # TODO: set later
        self.output_size = 1
        self.n_layers = input_dto.layers_pi
        self.hidden_size = input_dto.hidden_pi
        self.learn_rate = input_dto.lr_pi

        # build all layers
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for i in range(self.n_layers - 1)
            ]
        )
        self.layerN = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer1.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
        nn.init.constant_(self.layer1.bias, 0)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.layerN.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
        nn.init.constant_(self.layerN.bias, 0)

        # optimizer
        self.optimizer = optim.Adam(
            self.parameters(), lr=self.learn_rate
        )  # SGD or Adam
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    # # forward propagation
    # def forward(self, x):
    #     # normalize features with environment parameters
    #     x = normalize_features(x, self.env)

    #     loc = F.silu(self.layer1(x.squeeze()))

    #     for layer in self.hidden_layers:
    #         loc = F.silu(layer(loc))

    #     # output layer attempts
    #     loc = T.clamp(
    #         self.layerN(loc),
    #         min=-self.env.max_alpha,
    #         max=self.env.max_alpha,
    #     )

    #     # standard deviation of the Gaussian policy
    #     scale = T.tensor(0.03, device=self.device) ## what is this???

    #     return loc, scale
    
    def forward(self, x):
        x = x.squeeze(-1) if x.dim() > 2 else x    # safe reshape

        h = F.silu(self.layer1(x))
        for layer in self.hidden_layers:
            h = F.silu(layer(h))

        loc = self.layerN(h)
        loc = T.clamp(loc, -self.env.max_alpha, self.env.max_alpha)

        scale = T.tensor(0.03, device=self.device)
        return loc, scale


    

class CriticNN(nn.Module):
    """
    Value network aligned with PolicyNN constructor:
    CriticNN(input_dto: AlgoParams, env: BaseEnv)
    """
    def __init__(self, input_dto: AlgoParams, env: BaseEnv):
        super(CriticNN, self).__init__()

        # --- Mirror policy initialisation ---
        self.env = env
        self.input_size = 4                          # same placeholder as PolicyNN; update if needed
        self.output_size = 1
        self.n_layers = input_dto.layers_V           # critic has its own #layers parameter
        self.hidden_size = input_dto.hidden_V
        self.learn_rate = input_dto.lr_V

        # --- Build layers (same pattern as policy) ---
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.n_layers - 1)
            ]
        )
        self.layerN = nn.Linear(self.hidden_size, self.output_size)

        # --- Weight initialization (copy of policy) ---
        nn.init.normal_(self.layer1.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
        nn.init.constant_(self.layer1.bias, 0)

        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
            nn.init.constant_(layer.bias, 0)

        nn.init.normal_(self.layerN.weight, mean=0, std=1 / np.sqrt(self.input_size) / 2)
        nn.init.constant_(self.layerN.bias, 0)

        # --- Optimizer ---
        self.optimizer = optim.Adam(self.parameters(), lr=self.learn_rate)
        self.loss = nn.MSELoss()

        # --- Device management ---
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass:
        - Normalize features exactly like the policy
        - Same activation structure except final layer is linear (scalar value)
        """
        x = x.squeeze(-1) if x.dim() > 2 else x    # safe reshape

        h = F.silu(self.layer1(x))

        for layer in self.hidden_layers:
            h = F.silu(layer(h))

        value = self.layerN(h)  # no clamp; critic should remain unbounded

        return value
    


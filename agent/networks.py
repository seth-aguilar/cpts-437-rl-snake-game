"""
Neural Network architectures for Deep Q-Learning.

This module contains:
- DQN: Standard Deep Q-Network
- DuelingDQN: Dueling architecture that separates state value and advantage functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Standard Deep Q-Network.
    
    Architecture:
    - Input layer matching state dimension
    - Hidden layers with ReLU activation
    - Output layer with Q-values for each action
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize the DQN.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
        """
        super(DQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling Deep Q-Network.
    
    Separates the Q-value into:
    - V(s): State value function (how good is this state?)
    - A(s,a): Advantage function (how much better is this action than average?)
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
    
    This decomposition helps the network learn which states are valuable
    without having to learn the effect of each action for each state.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = None):
        """
        Initialize the Dueling DQN.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions
            hidden_dims: List of hidden layer dimensions (default: [128, 128])
        """
        super(DuelingDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
        
        self.action_dim = action_dim
        
        # Shared feature extraction layers
        feature_layers = []
        prev_dim = state_dim
        
        for i, hidden_dim in enumerate(hidden_dims[:-1]):
            feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_layer = nn.Sequential(*feature_layers) if feature_layers else nn.Identity()
        
        # If we have at least one hidden layer, use it
        if len(hidden_dims) > 0:
            feature_out_dim = hidden_dims[-2] if len(hidden_dims) > 1 else state_dim
            last_hidden = hidden_dims[-1]
        else:
            feature_out_dim = state_dim
            last_hidden = 64
        
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(feature_out_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, 1)
        )
        
        # Advantage stream: estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_out_dim, last_hidden),
            nn.ReLU(),
            nn.Linear(last_hidden, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Returns Q-values computed as: V(s) + A(s,a) - mean(A(s,a'))
        """
        features = self.feature_layer(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Subtract mean advantage for identifiability
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class ConvDQN(nn.Module):
    """
    Convolutional DQN for image-based input (full board state).
    
    Use this when representing the state as a 2D grid image.
    """
    
    def __init__(self, grid_height: int, grid_width: int, in_channels: int, 
                 action_dim: int):
        """
        Initialize the Convolutional DQN.
        
        Args:
            grid_height: Height of the game grid
            grid_width: Width of the game grid
            in_channels: Number of input channels (e.g., 3 for RGB or stacked frames)
            action_dim: Number of possible actions
        """
        super(ConvDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate the size after convolutions
        conv_out_size = 64 * grid_height * grid_width
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        conv_out = self.conv(x)
        flattened = conv_out.view(conv_out.size(0), -1)
        return self.fc(flattened)


class NoisyLinear(nn.Module):
    """
    Noisy Linear layer for exploration.
    
    Adds parametric noise to weights for exploration instead of
    epsilon-greedy. The noise parameters are learned during training.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / (self.in_features ** 0.5)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / (self.in_features ** 0.5))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / (self.out_features ** 0.5))
    
    def reset_noise(self):
        """Reset noise for exploration."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise using factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

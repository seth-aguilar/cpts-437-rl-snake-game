"""
Deep Q-Network Agent for Snake Game.

Implements:
- Standard DQN
- Double DQN (for reduced overestimation)
- Dueling DQN architecture
- Epsilon-greedy exploration with decay

The agent learns to play Snake by interacting with the environment,
storing experiences in a replay buffer, and updating its Q-network
using mini-batch gradient descent.
"""

import os
import random
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agent.networks import DQN, DuelingDQN
from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Deep Q-Network Agent with support for various DQN variants.
    
    Features:
    - Double DQN: Uses online network to select actions, target network to evaluate
    - Dueling DQN: Separates state value and advantage estimation
    - Prioritized Experience Replay: Samples important transitions more often
    - Epsilon-greedy exploration with configurable decay
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = None,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        device: str = None,
    ):
        """
        Initialize the DQN Agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions (3 for Snake: straight, left, right)
            hidden_dims: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay factor for epsilon per episode
            buffer_size: Maximum size of replay buffer
            batch_size: Number of transitions to sample for each update
            target_update_freq: How often to update target network (in steps)
            use_double_dqn: Whether to use Double DQN
            use_dueling: Whether to use Dueling DQN architecture
            use_prioritized_replay: Whether to use Prioritized Experience Replay
            device: Device to run on ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        # Initialize networks
        NetworkClass = DuelingDQN if use_dueling else DQN
        
        self.policy_net = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = NetworkClass(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss for stability
        
        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Tracking
        self.steps_done = 0
        self.episodes_done = 0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: If False, always exploit (for evaluation)
            
        Returns:
            Selected action (0: straight, 1: left, 2: right)
        """
        if training and random.random() < self.epsilon:
            # Explore: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploit: best action according to Q-network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one step of optimization on the Q-network.
        
        Returns:
            Loss value if update was performed, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample from replay buffer
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select actions, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use max Q from target network
                next_q = self.target_net(next_states).max(dim=1)[0]
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        td_errors = current_q - target_q
        
        if self.use_prioritized_replay:
            # Weighted loss for prioritized replay
            loss = (weights * td_errors.pow(2)).mean()
            # Update priorities
            priorities = td_errors.abs().detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        else:
            loss = self.loss_fn(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        self.optimizer.step()
        
        # Update step counter
        self.steps_done += 1
        
        # Update target network periodically
        if self.steps_done % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1
    
    def get_epsilon(self) -> float:
        """Get current exploration rate."""
        return self.epsilon
    
    def reset_epsilon(self):
        """Reset epsilon to starting value."""
        self.epsilon = self.epsilon_start
    
    def save(self, filepath: str):
        """
        Save agent state to file.
        
        Args:
            filepath: Path to save the checkpoint
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'use_double_dqn': self.use_double_dqn,
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to the checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']
        
        print(f"Agent loaded from {filepath}")
        print(f"  Episodes: {self.episodes_done}, Steps: {self.steps_done}, Epsilon: {self.epsilon:.4f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current agent statistics."""
        return {
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'buffer_size': len(self.replay_buffer),
        }

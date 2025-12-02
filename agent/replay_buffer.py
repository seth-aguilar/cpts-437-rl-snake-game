"""
Experience Replay Buffer for DQN training.

Stores transitions (state, action, reward, next_state, done) and allows
sampling mini-batches for training. This breaks correlations between
consecutive samples and makes learning more stable.
"""

import random
from collections import deque, namedtuple
from typing import Tuple, List

import numpy as np
import torch


# Named tuple for storing transitions
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Standard Experience Replay Buffer.
    
    Stores transitions and samples uniformly at random for training.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions with probability proportional to their TD-error,
    giving more training on surprising/important transitions.
    
    Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_frames: int = 100000):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight
            beta_frames: Number of frames over which to anneal beta to 1
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
    
    @property
    def beta(self) -> float:
        """Calculate current beta value (annealed over time)."""
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state after action
            done: Whether episode ended
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        transition = Transition(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones, indices, weights)
        """
        n = len(self.buffer)
        
        # Compute sampling probabilities
        priorities = self.priorities[:n]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        
        # Compute importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        weights = torch.FloatTensor(weights)
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        batch = Transition(*zip(*transitions))
        
        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.LongTensor(batch.action)
        rewards = torch.FloatTensor(batch.reward)
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(batch.done)
        
        self.frame += 1
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of sampled transitions
            priorities: New priority values (typically |TD-error| + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero priority
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

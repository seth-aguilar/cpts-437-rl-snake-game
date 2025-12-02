"""
Training metrics logger and visualization.

Tracks and visualizes:
- Episode scores
- Snake lengths
- Survival times
- Loss values
- Epsilon decay
- Learning progress over time
"""

import os
import json
from collections import deque
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np


class MetricsLogger:
    """
    Logger for tracking training metrics.
    
    Tracks:
    - Episode rewards
    - Scores (food eaten)
    - Snake lengths
    - Episode lengths (survival time)
    - Training loss
    - Epsilon values
    """
    
    def __init__(self, log_dir: str = "logs", window_size: int = 100):
        """
        Initialize the metrics logger.
        
        Args:
            log_dir: Directory to save logs
            window_size: Window size for moving averages
        """
        self.log_dir = log_dir
        self.window_size = window_size
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Episode metrics
        self.episode_rewards: List[float] = []
        self.episode_scores: List[int] = []
        self.episode_lengths: List[int] = []
        self.episode_snake_lengths: List[int] = []
        
        # Training metrics
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        
        # Evaluation metrics
        self.eval_scores: List[Dict[str, float]] = []
        
        # Moving averages
        self.reward_window = deque(maxlen=window_size)
        self.score_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        
        # Timestamp
        self.start_time = datetime.now()
        self.run_id = self.start_time.strftime("%Y%m%d_%H%M%S")
    
    def log_episode(self, reward: float, score: int, length: int, 
                    snake_length: int, epsilon: float):
        """
        Log metrics for a completed episode.
        
        Args:
            reward: Total episode reward
            score: Score (food eaten)
            length: Episode length (steps survived)
            snake_length: Final snake length
            epsilon: Current exploration rate
        """
        self.episode_rewards.append(reward)
        self.episode_scores.append(score)
        self.episode_lengths.append(length)
        self.episode_snake_lengths.append(snake_length)
        self.epsilons.append(epsilon)
        
        # Update moving averages
        self.reward_window.append(reward)
        self.score_window.append(score)
        self.length_window.append(length)
    
    def log_loss(self, loss: float):
        """Log training loss."""
        self.losses.append(loss)
    
    def log_evaluation(self, episode: int, avg_score: float, avg_length: float,
                       max_score: int, min_score: int):
        """Log evaluation results."""
        self.eval_scores.append({
            'episode': episode,
            'avg_score': avg_score,
            'avg_length': avg_length,
            'max_score': max_score,
            'min_score': min_score,
        })
    
    def get_moving_averages(self) -> Dict[str, float]:
        """Get current moving averages."""
        return {
            'avg_reward': np.mean(self.reward_window) if self.reward_window else 0,
            'avg_score': np.mean(self.score_window) if self.score_window else 0,
            'avg_length': np.mean(self.length_window) if self.length_window else 0,
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        n_episodes = len(self.episode_scores)
        
        if n_episodes == 0:
            return {'episodes': 0}
        
        return {
            'episodes': n_episodes,
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_score': np.mean(self.episode_scores),
            'max_score': max(self.episode_scores),
            'avg_length': np.mean(self.episode_lengths),
            'max_length': max(self.episode_lengths),
            'final_epsilon': self.epsilons[-1] if self.epsilons else 1.0,
            'moving_avg_reward': self.get_moving_averages()['avg_reward'],
            'moving_avg_score': self.get_moving_averages()['avg_score'],
        }
    
    def print_progress(self, episode: int, extra_info: str = ""):
        """Print training progress."""
        avgs = self.get_moving_averages()
        
        if self.episode_scores:
            print(f"Episode {episode:5d} | "
                  f"Score: {self.episode_scores[-1]:3d} | "
                  f"Avg Score: {avgs['avg_score']:6.2f} | "
                  f"Avg Length: {avgs['avg_length']:7.1f} | "
                  f"Epsilon: {self.epsilons[-1]:.4f}"
                  f"{' | ' + extra_info if extra_info else ''}")
    
    def save(self, filepath: Optional[str] = None):
        """Save metrics to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.log_dir, f"metrics_{self.run_id}.json")
        
        data = {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat(),
            'episode_rewards': self.episode_rewards,
            'episode_scores': self.episode_scores,
            'episode_lengths': self.episode_lengths,
            'episode_snake_lengths': self.episode_snake_lengths,
            'epsilons': self.epsilons,
            'eval_scores': self.eval_scores,
            'summary': self.get_summary(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath: str) -> 'MetricsLogger':
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger = cls()
        logger.run_id = data.get('run_id', 'loaded')
        logger.episode_rewards = data.get('episode_rewards', [])
        logger.episode_scores = data.get('episode_scores', [])
        logger.episode_lengths = data.get('episode_lengths', [])
        logger.episode_snake_lengths = data.get('episode_snake_lengths', [])
        logger.epsilons = data.get('epsilons', [])
        logger.eval_scores = data.get('eval_scores', [])
        
        return logger


def plot_training_progress(logger: MetricsLogger, save_path: Optional[str] = None):
    """
    Plot training progress.
    
    Args:
        logger: MetricsLogger with training data
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = range(1, len(logger.episode_scores) + 1)
    
    # Plot scores
    ax = axes[0, 0]
    ax.plot(episodes, logger.episode_scores, alpha=0.3, color='blue', label='Score')
    if len(logger.episode_scores) > 10:
        window = min(100, len(logger.episode_scores) // 10)
        moving_avg = np.convolve(logger.episode_scores, 
                                  np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(logger.episode_scores) + 1), 
                moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Score')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot episode lengths (survival time)
    ax = axes[0, 1]
    ax.plot(episodes, logger.episode_lengths, alpha=0.3, color='green', label='Length')
    if len(logger.episode_lengths) > 10:
        window = min(100, len(logger.episode_lengths) // 10)
        moving_avg = np.convolve(logger.episode_lengths,
                                  np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(logger.episode_lengths) + 1),
                moving_avg, color='darkgreen', linewidth=2, label=f'Moving Avg ({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('Survival Time (Episode Length)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    ax = axes[1, 0]
    ax.plot(episodes, logger.epsilons, color='orange')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (Epsilon)')
    ax.grid(True, alpha=0.3)
    
    # Plot evaluation scores
    ax = axes[1, 1]
    if logger.eval_scores:
        eval_episodes = [e['episode'] for e in logger.eval_scores]
        eval_avg_scores = [e['avg_score'] for e in logger.eval_scores]
        eval_max_scores = [e['max_score'] for e in logger.eval_scores]
        
        ax.plot(eval_episodes, eval_avg_scores, 'o-', color='blue', label='Avg Score')
        ax.plot(eval_episodes, eval_max_scores, 's--', color='green', label='Max Score')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Score')
        ax.set_title('Evaluation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No evaluation data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Evaluation Performance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_comparison(loggers: Dict[str, MetricsLogger], metric: str = 'score',
                   save_path: Optional[str] = None):
    """
    Plot comparison between different training runs.
    
    Args:
        loggers: Dictionary mapping run names to MetricsLogger objects
        metric: Which metric to compare ('score', 'length', 'reward')
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plots.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_map = {
        'score': 'episode_scores',
        'length': 'episode_lengths',
        'reward': 'episode_rewards',
    }
    
    for name, logger in loggers.items():
        data = getattr(logger, metric_map.get(metric, 'episode_scores'))
        episodes = range(1, len(data) + 1)
        
        # Plot moving average
        if len(data) > 10:
            window = min(100, len(data) // 10)
            moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(data) + 1), moving_avg, linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

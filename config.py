"""
Configuration management for the Snake RL project.

Contains default hyperparameters and configuration options for:
- Environment settings
- Agent hyperparameters
- Training parameters
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import json
import os


@dataclass
class EnvConfig:
    """Environment configuration."""
    grid_size: Tuple[int, int] = (20, 20)
    cell_size: int = 20
    step_penalty: float = -0.01
    food_reward: float = 10.0
    death_penalty: float = -10.0
    max_steps_without_food: int = 200
    render_mode: bool = False


@dataclass
class AgentConfig:
    """Agent hyperparameters."""
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.997  # Faster decay (was 0.995)
    buffer_size: int = 50000      # Smaller buffer (was 100000)
    batch_size: int = 64
    target_update_freq: int = 100
    use_double_dqn: bool = True
    use_dueling: bool = False
    use_prioritized_replay: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 1000
    max_steps_per_episode: int = 10000
    save_freq: int = 100
    eval_freq: int = 50
    eval_episodes: int = 10
    log_freq: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    render_training: bool = False
    render_eval: bool = False
    seed: Optional[int] = None


@dataclass
class Config:
    """Complete configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        config_dict = {
            'env': {
                'grid_size': list(self.env.grid_size),
                'cell_size': self.env.cell_size,
                'step_penalty': self.env.step_penalty,
                'food_reward': self.env.food_reward,
                'death_penalty': self.env.death_penalty,
                'max_steps_without_food': self.env.max_steps_without_food,
                'render_mode': self.env.render_mode,
            },
            'agent': {
                'hidden_dims': self.agent.hidden_dims,
                'learning_rate': self.agent.learning_rate,
                'gamma': self.agent.gamma,
                'epsilon_start': self.agent.epsilon_start,
                'epsilon_end': self.agent.epsilon_end,
                'epsilon_decay': self.agent.epsilon_decay,
                'buffer_size': self.agent.buffer_size,
                'batch_size': self.agent.batch_size,
                'target_update_freq': self.agent.target_update_freq,
                'use_double_dqn': self.agent.use_double_dqn,
                'use_dueling': self.agent.use_dueling,
                'use_prioritized_replay': self.agent.use_prioritized_replay,
            },
            'training': {
                'num_episodes': self.training.num_episodes,
                'max_steps_per_episode': self.training.max_steps_per_episode,
                'save_freq': self.training.save_freq,
                'eval_freq': self.training.eval_freq,
                'eval_episodes': self.training.eval_episodes,
                'log_freq': self.training.log_freq,
                'checkpoint_dir': self.training.checkpoint_dir,
                'log_dir': self.training.log_dir,
                'render_training': self.training.render_training,
                'render_eval': self.training.render_eval,
                'seed': self.training.seed,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Load env config
        if 'env' in config_dict:
            env_cfg = config_dict['env']
            config.env = EnvConfig(
                grid_size=tuple(env_cfg.get('grid_size', [20, 20])),
                cell_size=env_cfg.get('cell_size', 20),
                step_penalty=env_cfg.get('step_penalty', -0.01),
                food_reward=env_cfg.get('food_reward', 10.0),
                death_penalty=env_cfg.get('death_penalty', -10.0),
                max_steps_without_food=env_cfg.get('max_steps_without_food', 200),
                render_mode=env_cfg.get('render_mode', False),
            )
        
        # Load agent config
        if 'agent' in config_dict:
            agent_cfg = config_dict['agent']
            config.agent = AgentConfig(
                hidden_dims=agent_cfg.get('hidden_dims', [256, 256]),
                learning_rate=agent_cfg.get('learning_rate', 1e-3),
                gamma=agent_cfg.get('gamma', 0.99),
                epsilon_start=agent_cfg.get('epsilon_start', 1.0),
                epsilon_end=agent_cfg.get('epsilon_end', 0.01),
                epsilon_decay=agent_cfg.get('epsilon_decay', 0.995),
                buffer_size=agent_cfg.get('buffer_size', 100000),
                batch_size=agent_cfg.get('batch_size', 64),
                target_update_freq=agent_cfg.get('target_update_freq', 100),
                use_double_dqn=agent_cfg.get('use_double_dqn', True),
                use_dueling=agent_cfg.get('use_dueling', False),
                use_prioritized_replay=agent_cfg.get('use_prioritized_replay', False),
            )
        
        # Load training config
        if 'training' in config_dict:
            train_cfg = config_dict['training']
            config.training = TrainingConfig(
                num_episodes=train_cfg.get('num_episodes', 1000),
                max_steps_per_episode=train_cfg.get('max_steps_per_episode', 10000),
                save_freq=train_cfg.get('save_freq', 100),
                eval_freq=train_cfg.get('eval_freq', 50),
                eval_episodes=train_cfg.get('eval_episodes', 10),
                log_freq=train_cfg.get('log_freq', 10),
                checkpoint_dir=train_cfg.get('checkpoint_dir', 'checkpoints'),
                log_dir=train_cfg.get('log_dir', 'logs'),
                render_training=train_cfg.get('render_training', False),
                render_eval=train_cfg.get('render_eval', False),
                seed=train_cfg.get('seed', None),
            )
        
        print(f"Configuration loaded from {filepath}")
        return config


# Predefined configurations for different scenarios

def get_fast_training_config() -> Config:
    """Configuration optimized for fast experimentation."""
    config = Config()
    config.env.grid_size = (10, 10)
    config.training.num_episodes = 500
    config.agent.epsilon_decay = 0.99
    config.training.save_freq = 50
    config.training.eval_freq = 25
    return config


def get_standard_config() -> Config:
    """Standard training configuration."""
    return Config()


def get_intensive_config() -> Config:
    """Configuration for thorough training."""
    config = Config()
    config.training.num_episodes = 5000
    config.agent.epsilon_decay = 0.998
    config.agent.buffer_size = 200000
    config.training.save_freq = 200
    config.training.eval_freq = 100
    config.training.eval_episodes = 20
    return config


def get_dueling_config() -> Config:
    """Configuration using Dueling DQN."""
    config = Config()
    config.agent.use_dueling = True
    config.agent.hidden_dims = [256, 256]
    return config


def get_prioritized_config() -> Config:
    """Configuration using Prioritized Experience Replay."""
    config = Config()
    config.agent.use_prioritized_replay = True
    config.agent.learning_rate = 6.25e-5  # Lower LR for PER
    return config

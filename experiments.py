#!/usr/bin/env python3
"""
Hyperparameter tuning and experiment management.

This script allows running multiple training experiments with different
configurations to find the best hyperparameters.

Usage:
    python experiments.py --experiment baseline
    python experiments.py --experiment compare_architectures
    python experiments.py --experiment grid_search
"""

import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Any

import numpy as np

from train import train, evaluate_agent
from config import Config, EnvConfig, AgentConfig, TrainingConfig
from utils.metrics import MetricsLogger


def run_experiment(name: str, configs: List[Dict[str, Any]], 
                   num_runs: int = 1) -> Dict[str, Any]:
    """
    Run an experiment with multiple configurations.
    
    Args:
        name: Experiment name
        configs: List of configuration dictionaries
        num_runs: Number of runs per configuration
        
    Returns:
        Dictionary with experiment results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{name}_{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    results = {
        'experiment_name': name,
        'timestamp': timestamp,
        'num_configs': len(configs),
        'num_runs': num_runs,
        'configs': [],
        'results': []
    }
    
    for i, config_dict in enumerate(configs):
        config_name = config_dict.get('name', f'config_{i}')
        print(f"\n{'='*60}")
        print(f"Running configuration: {config_name}")
        print(f"{'='*60}")
        
        config_results = {
            'name': config_name,
            'config': config_dict,
            'runs': []
        }
        
        for run in range(num_runs):
            print(f"\n--- Run {run + 1}/{num_runs} ---")
            
            # Create config
            config = Config()
            
            # Apply configuration overrides
            if 'env' in config_dict:
                for key, value in config_dict['env'].items():
                    if hasattr(config.env, key):
                        setattr(config.env, key, value)
            
            if 'agent' in config_dict:
                for key, value in config_dict['agent'].items():
                    if hasattr(config.agent, key):
                        setattr(config.agent, key, value)
            
            if 'training' in config_dict:
                for key, value in config_dict['training'].items():
                    if hasattr(config.training, key):
                        setattr(config.training, key, value)
            
            # Set directories
            config.training.checkpoint_dir = f"{experiment_dir}/{config_name}/run_{run}/checkpoints"
            config.training.log_dir = f"{experiment_dir}/{config_name}/run_{run}/logs"
            
            # Train
            agent, logger = train(config)
            
            # Get final metrics
            summary = logger.get_summary()
            
            config_results['runs'].append({
                'run': run,
                'final_avg_score': summary['avg_score'],
                'max_score': summary['max_score'],
                'avg_length': summary['avg_length'],
                'moving_avg_score': summary['moving_avg_score'],
            })
        
        # Calculate aggregate statistics
        scores = [r['final_avg_score'] for r in config_results['runs']]
        config_results['aggregate'] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': max(r['max_score'] for r in config_results['runs']),
        }
        
        results['results'].append(config_results)
    
    # Save results
    results_path = f"{experiment_dir}/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {results_path}")
    
    # Print summary
    print("\nSummary:")
    print("-" * 60)
    for result in results['results']:
        print(f"{result['name']}: "
              f"Score = {result['aggregate']['mean_score']:.2f} ± {result['aggregate']['std_score']:.2f}, "
              f"Max = {result['aggregate']['max_score']}")
    
    return results


def baseline_experiment():
    """Run baseline experiment with default configuration."""
    configs = [
        {
            'name': 'baseline',
            'training': {
                'num_episodes': 500,
            }
        }
    ]
    return run_experiment('baseline', configs, num_runs=3)


def compare_architectures_experiment():
    """Compare DQN, Double DQN, and Dueling DQN."""
    configs = [
        {
            'name': 'standard_dqn',
            'agent': {
                'use_double_dqn': False,
                'use_dueling': False,
            },
            'training': {
                'num_episodes': 500,
            }
        },
        {
            'name': 'double_dqn',
            'agent': {
                'use_double_dqn': True,
                'use_dueling': False,
            },
            'training': {
                'num_episodes': 500,
            }
        },
        {
            'name': 'dueling_dqn',
            'agent': {
                'use_double_dqn': False,
                'use_dueling': True,
            },
            'training': {
                'num_episodes': 500,
            }
        },
        {
            'name': 'double_dueling_dqn',
            'agent': {
                'use_double_dqn': True,
                'use_dueling': True,
            },
            'training': {
                'num_episodes': 500,
            }
        },
    ]
    return run_experiment('architecture_comparison', configs, num_runs=2)


def learning_rate_experiment():
    """Grid search over learning rates."""
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]
    
    configs = [
        {
            'name': f'lr_{lr}',
            'agent': {
                'learning_rate': lr,
            },
            'training': {
                'num_episodes': 400,
            }
        }
        for lr in learning_rates
    ]
    return run_experiment('learning_rate_search', configs, num_runs=2)


def epsilon_decay_experiment():
    """Test different epsilon decay rates."""
    decay_rates = [0.99, 0.995, 0.998, 0.999]
    
    configs = [
        {
            'name': f'decay_{rate}',
            'agent': {
                'epsilon_decay': rate,
            },
            'training': {
                'num_episodes': 500,
            }
        }
        for rate in decay_rates
    ]
    return run_experiment('epsilon_decay_search', configs, num_runs=2)


def grid_size_experiment():
    """Test different grid sizes."""
    grid_sizes = [(10, 10), (15, 15), (20, 20), (25, 25)]
    
    configs = [
        {
            'name': f'grid_{size[0]}x{size[1]}',
            'env': {
                'grid_size': size,
            },
            'training': {
                'num_episodes': 500,
            }
        }
        for size in grid_sizes
    ]
    return run_experiment('grid_size_comparison', configs, num_runs=2)


def network_size_experiment():
    """Test different network architectures."""
    hidden_dims_list = [
        [64, 64],
        [128, 128],
        [256, 256],
        [256, 128, 64],
        [512, 256],
    ]
    
    configs = [
        {
            'name': f'net_{"_".join(map(str, dims))}',
            'agent': {
                'hidden_dims': dims,
            },
            'training': {
                'num_episodes': 400,
            }
        }
        for dims in hidden_dims_list
    ]
    return run_experiment('network_size_search', configs, num_runs=2)


EXPERIMENTS = {
    'baseline': baseline_experiment,
    'compare_architectures': compare_architectures_experiment,
    'learning_rate': learning_rate_experiment,
    'epsilon_decay': epsilon_decay_experiment,
    'grid_size': grid_size_experiment,
    'network_size': network_size_experiment,
}


def main():
    parser = argparse.ArgumentParser(description="Run RL experiments")
    
    parser.add_argument("--experiment", type=str, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help="Which experiment to run")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available experiments:")
        for name in EXPERIMENTS:
            print(f"  - {name}")
        return
    
    experiment_fn = EXPERIMENTS[args.experiment]
    experiment_fn()


if __name__ == "__main__":
    main()

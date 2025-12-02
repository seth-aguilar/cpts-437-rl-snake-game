#!/usr/bin/env python3
"""
Evaluation script for trained Snake RL agent.

This script loads a trained model and evaluates its performance,
optionally with visualization.

Usage:
    python evaluate.py checkpoints/best_model.pt
    python evaluate.py checkpoints/best_model.pt --episodes 100 --render
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np

from env.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent
from config import Config


def evaluate(model_path: str, config: Optional[Config] = None,
             num_episodes: int = 100, render: bool = False,
             fps: int = 15, verbose: bool = True) -> dict:
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to trained model checkpoint
        config: Environment configuration (optional)
        num_episodes: Number of evaluation episodes
        render: Whether to render the game
        fps: Frames per second for rendering
        verbose: Whether to print episode details
        
    Returns:
        Dictionary with evaluation metrics
    """
    if config is None:
        config = Config()
    
    # Initialize environment
    env = SnakeEnv(
        grid_size=config.env.grid_size,
        cell_size=config.env.cell_size,
        step_penalty=config.env.step_penalty,
        food_reward=config.env.food_reward,
        death_penalty=config.env.death_penalty,
        max_steps_without_food=config.env.max_steps_without_food,
        render_mode=render,
    )
    
    # Get dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 3
    
    # Initialize and load agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.agent.hidden_dims,
        use_double_dqn=config.agent.use_double_dqn,
        use_dueling=config.agent.use_dueling,
    )
    
    agent.load(model_path)
    
    # Evaluation metrics
    scores = []
    lengths = []
    rewards = []
    deaths_by_collision = 0
    deaths_by_stuck = 0
    
    print(f"\nEvaluating agent over {num_episodes} episodes...")
    print("-" * 50)
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if render:
                env.render(fps=fps)
        
        scores.append(env.score)
        lengths.append(steps)
        rewards.append(episode_reward)
        
        if info.get('reason') == 'collision':
            deaths_by_collision += 1
        elif info.get('reason') == 'stuck':
            deaths_by_stuck += 1
        
        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:4d}: Score = {env.score:3d}, "
                  f"Length = {steps:4d}, Reward = {episode_reward:.2f}")
    
    env.close()
    
    # Calculate statistics
    results = {
        'num_episodes': num_episodes,
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'median_score': np.median(scores),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'max_length': max(lengths),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'deaths_by_collision': deaths_by_collision,
        'deaths_by_stuck': deaths_by_stuck,
        'collision_rate': deaths_by_collision / num_episodes,
        'stuck_rate': deaths_by_stuck / num_episodes,
    }
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    
    print(f"\nScore Statistics:")
    print(f"  Average Score: {results['avg_score']:.2f} ± {results['std_score']:.2f}")
    print(f"  Max Score:     {results['max_score']}")
    print(f"  Min Score:     {results['min_score']}")
    print(f"  Median Score:  {results['median_score']:.1f}")
    
    print(f"\nSurvival Statistics:")
    print(f"  Average Length: {results['avg_length']:.1f} ± {results['std_length']:.1f} steps")
    print(f"  Max Length:     {results['max_length']} steps")
    
    print(f"\nReward Statistics:")
    print(f"  Average Reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    
    print(f"\nDeath Analysis:")
    print(f"  Deaths by Collision: {results['deaths_by_collision']} ({results['collision_rate']*100:.1f}%)")
    print(f"  Deaths by Getting Stuck: {results['deaths_by_stuck']} ({results['stuck_rate']*100:.1f}%)")
    
    print("=" * 50)


def compare_agents(model_paths: list, names: list = None, 
                   num_episodes: int = 50, config: Optional[Config] = None):
    """
    Compare multiple trained agents.
    
    Args:
        model_paths: List of paths to model checkpoints
        names: Names for each model (optional)
        num_episodes: Number of episodes for each evaluation
        config: Environment configuration
    """
    if names is None:
        names = [f"Model {i+1}" for i in range(len(model_paths))]
    
    results = {}
    
    for name, path in zip(names, model_paths):
        print(f"\nEvaluating {name}...")
        results[name] = evaluate(path, config=config, num_episodes=num_episodes, 
                                 render=False, verbose=False)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Avg Score':>12} {'Max Score':>12} {'Avg Length':>12}")
    print("-" * 70)
    
    for name, res in results.items():
        print(f"{name:<20} {res['avg_score']:>10.2f}±{res['std_score']:<5.2f} "
              f"{res['max_score']:>10} {res['avg_length']:>12.1f}")
    
    print("=" * 70)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Snake RL Agent")
    
    parser.add_argument("model", type=str, nargs='?', default="checkpoints/best_model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Render the game during evaluation")
    parser.add_argument("--fps", type=int, default=15,
                        help="FPS for rendering")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--compare", type=str, nargs='+', default=None,
                        help="Compare multiple models")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Override grid size")
    
    args = parser.parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    else:
        config = Config()
    
    if args.grid_size:
        config.env.grid_size = (args.grid_size, args.grid_size)
    
    # Compare models or evaluate single model
    if args.compare:
        compare_agents(args.compare, num_episodes=args.episodes, config=config)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            print("Train a model first using: python train.py")
            sys.exit(1)
        
        results = evaluate(
            args.model,
            config=config,
            num_episodes=args.episodes,
            render=args.render,
            fps=args.fps
        )
        
        print_results(results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for the Snake RL agent.

This script trains a Deep Q-Network agent to play the Snake game.
It supports various configurations including:
- Standard DQN
- Double DQN
- Dueling DQN
- Prioritized Experience Replay

Usage:
    python train.py                     # Train with default config
    python train.py --episodes 2000     # Train for 2000 episodes
    python train.py --config config.json # Load config from file
    python train.py --render            # Train with rendering
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch

from env.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent
from config import Config, get_standard_config, get_fast_training_config
from utils.metrics import MetricsLogger, plot_training_progress


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_agent(env: SnakeEnv, agent: DQNAgent, num_episodes: int = 10,
                   render: bool = False) -> dict:
    """
    Evaluate the agent's performance.
    
    Args:
        env: Snake environment
        agent: Trained agent
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
        
    Returns:
        Dictionary with evaluation metrics
    """
    scores = []
    lengths = []
    
    original_render_mode = env.render_mode
    env.render_mode = render
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        episode_length = 0
        
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)
            episode_length += 1
            
            if render:
                env.render(fps=15)
        
        scores.append(env.score)
        lengths.append(episode_length)
    
    env.render_mode = original_render_mode
    
    return {
        'avg_score': np.mean(scores),
        'std_score': np.std(scores),
        'max_score': max(scores),
        'min_score': min(scores),
        'avg_length': np.mean(lengths),
        'std_length': np.std(lengths),
    }


def train(config: Config, resume_from: str = None):
    """
    Train the DQN agent.
    
    Args:
        config: Training configuration
        resume_from: Path to checkpoint to resume from
    """
    # Set seed for reproducibility
    if config.training.seed is not None:
        set_seed(config.training.seed)
        print(f"Random seed set to {config.training.seed}")
    
    # Create directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Initialize environment
    env = SnakeEnv(
        grid_size=config.env.grid_size,
        cell_size=config.env.cell_size,
        step_penalty=config.env.step_penalty,
        food_reward=config.env.food_reward,
        death_penalty=config.env.death_penalty,
        max_steps_without_food=config.env.max_steps_without_food,
        render_mode=config.training.render_training,
    )
    
    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 3  # straight, left, right
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Grid size: {config.env.grid_size}")
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config.agent.hidden_dims,
        learning_rate=config.agent.learning_rate,
        gamma=config.agent.gamma,
        epsilon_start=config.agent.epsilon_start,
        epsilon_end=config.agent.epsilon_end,
        epsilon_decay=config.agent.epsilon_decay,
        buffer_size=config.agent.buffer_size,
        batch_size=config.agent.batch_size,
        target_update_freq=config.agent.target_update_freq,
        use_double_dqn=config.agent.use_double_dqn,
        use_dueling=config.agent.use_dueling,
        use_prioritized_replay=config.agent.use_prioritized_replay,
    )
    
    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from and os.path.exists(resume_from):
        agent.load(resume_from)
        start_episode = agent.episodes_done
        print(f"Resuming training from episode {start_episode}")
    
    # Initialize metrics logger
    logger = MetricsLogger(log_dir=config.training.log_dir)
    
    # Training info
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print(f"Episodes: {config.training.num_episodes}")
    print(f"Double DQN: {config.agent.use_double_dqn}")
    print(f"Dueling DQN: {config.agent.use_dueling}")
    print(f"Prioritized Replay: {config.agent.use_prioritized_replay}")
    print(f"Hidden layers: {config.agent.hidden_dims}")
    print(f"Learning rate: {config.agent.learning_rate}")
    print(f"Gamma: {config.agent.gamma}")
    print(f"Epsilon decay: {config.agent.epsilon_decay}")
    print("="*60 + "\n")
    
    # Save initial config
    config_path = os.path.join(config.training.log_dir, f"config_{logger.run_id}.json")
    config.save(config_path)
    
    # Create training log file for batch outputs
    training_log_path = os.path.join(config.training.log_dir, f"training_log_{logger.run_id}.txt")
    
    # Write header to training log file
    with open(training_log_path, 'w') as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("SNAKE RL TRAINING LOG\n")
        log_file.write("=" * 70 + "\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Run ID: {logger.run_id}\n")
        log_file.write("-" * 70 + "\n")
        log_file.write("Configuration:\n")
        log_file.write(f"  Episodes: {config.training.num_episodes}\n")
        log_file.write(f"  Grid Size: {config.env.grid_size}\n")
        log_file.write(f"  Double DQN: {config.agent.use_double_dqn}\n")
        log_file.write(f"  Dueling DQN: {config.agent.use_dueling}\n")
        log_file.write(f"  Prioritized Replay: {config.agent.use_prioritized_replay}\n")
        log_file.write(f"  Hidden Layers: {config.agent.hidden_dims}\n")
        log_file.write(f"  Learning Rate: {config.agent.learning_rate}\n")
        log_file.write(f"  Gamma: {config.agent.gamma}\n")
        log_file.write(f"  Epsilon Decay: {config.agent.epsilon_decay}\n")
        log_file.write("=" * 70 + "\n\n")
        log_file.write("TRAINING PROGRESS\n")
        log_file.write("-" * 70 + "\n")
        log_file.write(f"{'Episode':>8} | {'Score':>6} | {'Avg Score':>10} | {'Avg Length':>10} | {'Epsilon':>8} | {'Loss':>10}\n")
        log_file.write("-" * 70 + "\n")
    
    print(f"Training log will be saved to: {training_log_path}")
    
    best_avg_score = 0
    
    # Training loop
    for episode in range(start_episode, config.training.num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        steps = 0
        
        while not done and steps < config.training.max_steps_per_episode:
            # Select action
            action = agent.select_action(state, training=True)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            # Render if enabled
            if config.training.render_training:
                env.render(fps=30)
            
            state = next_state
            episode_reward += reward
            steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Log episode metrics
        logger.log_episode(
            reward=episode_reward,
            score=env.score,
            length=steps,
            snake_length=len(env.snake),
            epsilon=agent.get_epsilon()
        )
        
        if episode_loss:
            avg_loss = np.mean(episode_loss)
            logger.log_loss(avg_loss)
        
        # Print progress
        if (episode + 1) % config.training.log_freq == 0:
            loss_str = f"Loss: {avg_loss:.4f}" if episode_loss else ""
            logger.print_progress(episode + 1, loss_str)
            
            # Write batch data to training log file
            avgs = logger.get_moving_averages()
            loss_val = avg_loss if episode_loss else 0.0
            with open(training_log_path, 'a') as log_file:
                log_file.write(f"{episode + 1:>8} | {env.score:>6} | {avgs['avg_score']:>10.2f} | {avgs['avg_length']:>10.1f} | {agent.get_epsilon():>8.4f} | {loss_val:>10.4f}\n")
        
        # Evaluate periodically
        if (episode + 1) % config.training.eval_freq == 0:
            eval_results = evaluate_agent(
                env, agent, 
                num_episodes=config.training.eval_episodes,
                render=config.training.render_eval
            )
            
            logger.log_evaluation(
                episode=episode + 1,
                avg_score=eval_results['avg_score'],
                avg_length=eval_results['avg_length'],
                max_score=eval_results['max_score'],
                min_score=eval_results['min_score']
            )
            
            print(f"\n  Evaluation ({config.training.eval_episodes} episodes):")
            print(f"    Avg Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}")
            print(f"    Max Score: {eval_results['max_score']}")
            print(f"    Avg Length: {eval_results['avg_length']:.1f}\n")
            
            # Write evaluation results to training log file
            with open(training_log_path, 'a') as log_file:
                log_file.write("\n" + "-" * 40 + "\n")
                log_file.write(f"EVALUATION at Episode {episode + 1}:\n")
                log_file.write(f"  Avg Score: {eval_results['avg_score']:.2f} ± {eval_results['std_score']:.2f}\n")
                log_file.write(f"  Max Score: {eval_results['max_score']}\n")
                log_file.write(f"  Min Score: {eval_results['min_score']}\n")
                log_file.write(f"  Avg Length: {eval_results['avg_length']:.1f}\n")
                log_file.write("-" * 40 + "\n\n")
            
            # Save best model
            if eval_results['avg_score'] > best_avg_score:
                best_avg_score = eval_results['avg_score']
                best_path = os.path.join(config.training.checkpoint_dir, "best_model.pt")
                agent.save(best_path)
                print(f"  New best model saved (avg score: {best_avg_score:.2f})\n")
                
                # Log best model save to file
                with open(training_log_path, 'a') as log_file:
                    log_file.write(f"*** NEW BEST MODEL SAVED (avg score: {best_avg_score:.2f}) ***\n\n")
        
        # Save checkpoint periodically
        if (episode + 1) % config.training.save_freq == 0:
            checkpoint_path = os.path.join(
                config.training.checkpoint_dir, 
                f"checkpoint_ep{episode+1}.pt"
            )
            agent.save(checkpoint_path)
    
    # Training complete
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_eval = evaluate_agent(env, agent, num_episodes=20)
    print(f"  Avg Score: {final_eval['avg_score']:.2f} ± {final_eval['std_score']:.2f}")
    print(f"  Max Score: {final_eval['max_score']}")
    print(f"  Avg Length: {final_eval['avg_length']:.1f}")
    
    # Save final model
    final_path = os.path.join(config.training.checkpoint_dir, "final_model.pt")
    agent.save(final_path)
    
    # Save metrics
    metrics_path = logger.save()
    
    # Print summary
    summary = logger.get_summary()
    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {summary['episodes']}")
    print(f"  Best Avg Score: {best_avg_score:.2f}")
    print(f"  Overall Avg Score: {summary['avg_score']:.2f}")
    print(f"  Max Score Achieved: {summary['max_score']}")
    
    # Write final summary to training log file
    with open(training_log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 70 + "\n")
        log_file.write("TRAINING COMPLETE\n")
        log_file.write("=" * 70 + "\n")
        log_file.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write("FINAL EVALUATION (20 episodes):\n")
        log_file.write(f"  Avg Score: {final_eval['avg_score']:.2f} ± {final_eval['std_score']:.2f}\n")
        log_file.write(f"  Max Score: {final_eval['max_score']}\n")
        log_file.write(f"  Min Score: {final_eval['min_score']}\n")
        log_file.write(f"  Avg Length: {final_eval['avg_length']:.1f}\n\n")
        log_file.write("TRAINING SUMMARY:\n")
        log_file.write(f"  Total Episodes: {summary['episodes']}\n")
        log_file.write(f"  Best Avg Score: {best_avg_score:.2f}\n")
        log_file.write(f"  Overall Avg Score: {summary['avg_score']:.2f}\n")
        log_file.write(f"  Max Score Achieved: {summary['max_score']}\n")
        log_file.write(f"  Average Episode Length: {summary['avg_length']:.1f}\n")
        log_file.write("=" * 70 + "\n")
    
    print(f"\nTraining log saved to: {training_log_path}")
    
    # Plot results
    try:
        plot_path = os.path.join(config.training.log_dir, f"training_plot_{logger.run_id}.png")
        plot_training_progress(logger, save_path=plot_path)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    # Cleanup
    env.close()
    
    return agent, logger


def main():
    parser = argparse.ArgumentParser(description="Train Snake RL Agent")
    
    parser.add_argument("--episodes", type=int, default=None,
                        help="Number of training episodes")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--render", action="store_true",
                        help="Render training")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast training config (smaller grid, fewer episodes)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--double-dqn", action="store_true", default=True,
                        help="Use Double DQN (default: True)")
    parser.add_argument("--dueling", action="store_true",
                        help="Use Dueling DQN architecture")
    parser.add_argument("--prioritized", action="store_true",
                        help="Use Prioritized Experience Replay")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Discount factor")
    parser.add_argument("--grid-size", type=int, default=None,
                        help="Grid size (square)")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and os.path.exists(args.config):
        config = Config.load(args.config)
    elif args.fast:
        config = get_fast_training_config()
        print("Using fast training configuration")
    else:
        config = get_standard_config()
    
    # Override config with command line arguments
    if args.episodes is not None:
        config.training.num_episodes = args.episodes
    if args.render:
        config.training.render_training = True
    if args.seed is not None:
        config.training.seed = args.seed
    if args.dueling:
        config.agent.use_dueling = True
    if args.prioritized:
        config.agent.use_prioritized_replay = True
    if args.lr is not None:
        config.agent.learning_rate = args.lr
    if args.gamma is not None:
        config.agent.gamma = args.gamma
    if args.grid_size is not None:
        config.env.grid_size = (args.grid_size, args.grid_size)
    
    # Train
    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()

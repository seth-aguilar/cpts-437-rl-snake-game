#!/usr/bin/env python3
"""
Curriculum Learning for Snake RL.

Implements progressive training where the agent starts with easier
tasks and gradually moves to harder ones. This can help stabilize
and accelerate training.

Curriculum stages:
1. Small grid, short timeout
2. Small grid, normal timeout
3. Medium grid, normal timeout
4. Full size grid, normal timeout
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

from env.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent
from config import Config


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    name: str
    grid_size: Tuple[int, int]
    max_steps_without_food: int
    episodes: int
    promotion_threshold: float  # Average score needed to advance


class CurriculumTrainer:
    """
    Curriculum learning trainer for Snake.
    
    Progressively increases difficulty as the agent improves.
    """
    
    def __init__(self, config: Config, stages: List[CurriculumStage] = None):
        """
        Initialize the curriculum trainer.
        
        Args:
            config: Base configuration
            stages: List of curriculum stages (uses default if None)
        """
        self.config = config
        
        if stages is None:
            self.stages = self._default_stages()
        else:
            self.stages = stages
        
        self.current_stage = 0
    
    def _default_stages(self) -> List[CurriculumStage]:
        """Create default curriculum stages."""
        return [
            CurriculumStage(
                name="Tiny Grid",
                grid_size=(8, 8),
                max_steps_without_food=100,
                episodes=150,
                promotion_threshold=2.5,
            ),
            CurriculumStage(
                name="Small Grid",
                grid_size=(12, 12),
                max_steps_without_food=150,
                episodes=200,
                promotion_threshold=4.0,
            ),
            CurriculumStage(
                name="Medium Grid",
                grid_size=(16, 16),
                max_steps_without_food=200,
                episodes=300,
                promotion_threshold=6.0,
            ),
            CurriculumStage(
                name="Full Grid",
                grid_size=(20, 20),
                max_steps_without_food=200,
                episodes=400,
                promotion_threshold=12.0,
            ),
        ]
    
    def get_stage_env(self, stage_idx: int) -> SnakeEnv:
        """Create environment for a specific stage."""
        stage = self.stages[stage_idx]
        
        return SnakeEnv(
            grid_size=stage.grid_size,
            cell_size=self.config.env.cell_size,
            step_penalty=self.config.env.step_penalty,
            food_reward=self.config.env.food_reward,
            death_penalty=self.config.env.death_penalty,
            max_steps_without_food=stage.max_steps_without_food,
            render_mode=self.config.training.render_training,
        )
    
    def should_promote(self, avg_score: float) -> bool:
        """Check if agent should advance to next stage."""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        threshold = self.stages[self.current_stage].promotion_threshold
        return avg_score >= threshold
    
    def promote(self):
        """Advance to the next curriculum stage."""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            print(f"\n*** Promoted to stage: {self.stages[self.current_stage].name} ***\n")
    
    def get_current_stage(self) -> CurriculumStage:
        """Get the current curriculum stage."""
        return self.stages[self.current_stage]
    
    def is_complete(self) -> bool:
        """Check if all stages are complete."""
        return self.current_stage >= len(self.stages) - 1


def train_with_curriculum(config: Config, stages: List[CurriculumStage] = None):
    """
    Train agent using curriculum learning.
    
    Args:
        config: Training configuration
        stages: Optional custom curriculum stages
    """
    import os
    import numpy as np
    from datetime import datetime
    from collections import deque
    from utils.metrics import MetricsLogger
    
    # Create directories
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # Create curriculum trainer
    curriculum = CurriculumTrainer(config, stages)
    
    # Initialize agent (state dim is same regardless of grid size)
    env = curriculum.get_stage_env(0)
    state = env.reset()
    state_dim = len(state)
    action_dim = 3
    
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
    )
    
    # Initialize logger
    logger = MetricsLogger(log_dir=config.training.log_dir)
    
    # Create training log file
    training_log_path = os.path.join(config.training.log_dir, f"curriculum_log_{logger.run_id}.txt")
    
    # Write header to training log file
    with open(training_log_path, 'w') as log_file:
        log_file.write("=" * 70 + "\n")
        log_file.write("CURRICULUM LEARNING TRAINING LOG\n")
        log_file.write("=" * 70 + "\n")
        log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Run ID: {logger.run_id}\n")
        log_file.write(f"State Dimension: {state_dim}\n")
        log_file.write("-" * 70 + "\n")
        log_file.write("Configuration:\n")
        log_file.write(f"  Total Episodes: {config.training.num_episodes}\n")
        log_file.write(f"  Double DQN: {config.agent.use_double_dqn}\n")
        log_file.write(f"  Dueling DQN: {config.agent.use_dueling}\n")
        log_file.write(f"  Hidden Layers: {config.agent.hidden_dims}\n")
        log_file.write(f"  Learning Rate: {config.agent.learning_rate}\n")
        log_file.write(f"  Gamma: {config.agent.gamma}\n")
        log_file.write(f"  Epsilon Decay: {config.agent.epsilon_decay}\n")
        log_file.write("-" * 70 + "\n")
        log_file.write("Curriculum Stages:\n")
        for i, stage in enumerate(curriculum.stages):
            log_file.write(f"  Stage {i+1}: {stage.name} ({stage.grid_size[0]}x{stage.grid_size[1]}) - "
                          f"Threshold: {stage.promotion_threshold}\n")
        log_file.write("=" * 70 + "\n\n")
    
    print("\n" + "=" * 60)
    print("CURRICULUM LEARNING")
    print("=" * 60)
    for i, stage in enumerate(curriculum.stages):
        print(f"  Stage {i+1}: {stage.name} ({stage.grid_size[0]}x{stage.grid_size[1]})")
    print("=" * 60)
    print(f"Training log: {training_log_path}\n")
    
    total_episodes = 0
    score_window = deque(maxlen=50)
    
    # Training loop through curriculum stages
    while not curriculum.is_complete():
        stage = curriculum.get_current_stage()
        env = curriculum.get_stage_env(curriculum.current_stage)
        
        print(f"\n--- Stage: {stage.name} ({stage.grid_size[0]}x{stage.grid_size[1]}) ---")
        print(f"Episodes: {stage.episodes}, Promotion threshold: {stage.promotion_threshold}")
        
        # Log stage start
        with open(training_log_path, 'a') as log_file:
            log_file.write(f"\n{'='*70}\n")
            log_file.write(f"STAGE: {stage.name} ({stage.grid_size[0]}x{stage.grid_size[1]})\n")
            log_file.write(f"{'='*70}\n")
            log_file.write(f"Episodes: {stage.episodes}, Promotion threshold: {stage.promotion_threshold}\n")
            log_file.write("-" * 70 + "\n")
            log_file.write(f"{'Episode':>8} | {'Score':>6} | {'Avg Score':>10} | {'Epsilon':>8}\n")
            log_file.write("-" * 70 + "\n")
        
        for ep in range(stage.episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            steps = 0
            
            while not done:
                action = agent.select_action(state, training=True)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                if config.training.render_training:
                    env.render()
            
            agent.decay_epsilon()
            
            score_window.append(env.score)
            avg_score = np.mean(score_window)
            
            logger.log_episode(
                reward=episode_reward,
                score=env.score,
                length=steps,
                snake_length=len(env.snake),
                epsilon=agent.get_epsilon()
            )
            
            total_episodes += 1
            
            if (ep + 1) % 20 == 0:
                print(f"  Episode {ep+1}/{stage.episodes} | "
                      f"Score: {env.score} | Avg: {avg_score:.2f} | "
                      f"ε: {agent.get_epsilon():.3f}")
                
                # Log to file
                with open(training_log_path, 'a') as log_file:
                    log_file.write(f"{ep+1:>8} | {env.score:>6} | {avg_score:>10.2f} | {agent.get_epsilon():>8.4f}\n")
            
            # Check for promotion
            if len(score_window) >= 50 and curriculum.should_promote(avg_score):
                print(f"\n  Promotion threshold reached! (avg: {avg_score:.2f})")
                
                # Log promotion
                with open(training_log_path, 'a') as log_file:
                    log_file.write("\n" + "*" * 50 + "\n")
                    log_file.write(f"PROMOTED! Avg score: {avg_score:.2f} >= {stage.promotion_threshold}\n")
                    log_file.write(f"Total episodes in stage: {ep+1}\n")
                    log_file.write("*" * 50 + "\n")
                
                curriculum.promote()
                score_window.clear()
                break
        
        env.close()
    
    # Final stage training
    print("\n--- Final Stage Training ---")
    final_stage = curriculum.stages[-1]
    env = curriculum.get_stage_env(len(curriculum.stages) - 1)
    
    remaining_episodes = max(0, config.training.num_episodes - total_episodes)
    
    # Log final stage start
    with open(training_log_path, 'a') as log_file:
        log_file.write(f"\n{'='*70}\n")
        log_file.write(f"FINAL STAGE TRAINING: {final_stage.name} ({final_stage.grid_size[0]}x{final_stage.grid_size[1]})\n")
        log_file.write(f"{'='*70}\n")
        log_file.write(f"Remaining episodes: {remaining_episodes}\n")
        log_file.write("-" * 70 + "\n")
        log_file.write(f"{'Episode':>8} | {'Avg Score':>10} | {'Avg Length':>10} | {'Epsilon':>8}\n")
        log_file.write("-" * 70 + "\n")
    
    for ep in range(remaining_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if config.training.render_training:
                env.render()
        
        agent.decay_epsilon()
        
        logger.log_episode(
            reward=episode_reward,
            score=env.score,
            length=steps,
            snake_length=len(env.snake),
            epsilon=agent.get_epsilon()
        )
        
        total_episodes += 1
        
        if (ep + 1) % 50 == 0:
            avgs = logger.get_moving_averages()
            print(f"  Episode {ep+1}/{remaining_episodes} | "
                  f"Avg Score: {avgs['avg_score']:.2f} | "
                  f"ε: {agent.get_epsilon():.3f}")
            
            # Log to file
            with open(training_log_path, 'a') as log_file:
                log_file.write(f"{ep+1:>8} | {avgs['avg_score']:>10.2f} | {avgs['avg_length']:>10.1f} | {agent.get_epsilon():>8.4f}\n")
    
    env.close()
    
    # Save final model
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    agent.save(os.path.join(config.training.checkpoint_dir, "curriculum_model.pt"))
    metrics_path = logger.save()
    
    # Write final summary to log file
    from datetime import datetime
    summary = logger.get_summary()
    with open(training_log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 70 + "\n")
        log_file.write("TRAINING COMPLETE\n")
        log_file.write("=" * 70 + "\n")
        log_file.write(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write("FINAL SUMMARY:\n")
        log_file.write(f"  Total Episodes: {total_episodes}\n")
        log_file.write(f"  Final Avg Score: {summary['moving_avg_score']:.2f}\n")
        log_file.write(f"  Overall Avg Score: {summary['avg_score']:.2f}\n")
        log_file.write(f"  Max Score Achieved: {summary['max_score']}\n")
        log_file.write(f"  Average Episode Length: {summary['avg_length']:.1f}\n")
        log_file.write(f"  Final Epsilon: {agent.get_epsilon():.4f}\n")
        log_file.write("-" * 70 + "\n")
        log_file.write(f"Model saved to: checkpoints/curriculum_model.pt\n")
        log_file.write(f"Metrics saved to: {metrics_path}\n")
        log_file.write("=" * 70 + "\n")
    
    print("\n" + "=" * 60)
    print("Curriculum Training Complete!")
    print("=" * 60)
    print(f"\nTraining Summary:")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Final Avg Score: {summary['moving_avg_score']:.2f}")
    print(f"  Overall Avg Score: {summary['avg_score']:.2f}")
    print(f"  Max Score Achieved: {summary['max_score']}")
    
    # Generate and save training plots
    from utils.metrics import plot_training_progress
    try:
        plot_path = os.path.join(config.training.log_dir, f"curriculum_plot_{logger.run_id}.png")
        plot_training_progress(logger, save_path=plot_path)
        print(f"\nTraining plot saved to: {plot_path}")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
    
    print(f"Training log saved to: {training_log_path}")
    print(f"Metrics saved to: {metrics_path}")
    print("=" * 60)
    
    return agent, logger


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train with curriculum learning")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Total episodes including curriculum")
    parser.add_argument("--render", action="store_true",
                        help="Render training")
    
    args = parser.parse_args()
    
    config = Config()
    config.training.num_episodes = args.episodes
    config.training.render_training = args.render
    
    train_with_curriculum(config)

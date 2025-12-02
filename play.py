#!/usr/bin/env python3
"""
Demo script for watching a trained Snake agent play.

This script provides an interactive way to watch the trained agent
play the Snake game with visualization.

Usage:
    python play.py                          # Watch trained agent play
    python play.py checkpoints/best_model.pt  # Use specific model
    python play.py --human                  # Play manually
"""

import argparse
import os
import sys

import pygame

from env.snake_env import SnakeEnv
from agent.dqn_agent import DQNAgent
from config import Config


def watch_agent(model_path: str, config: Config = None, 
                num_games: int = 5, fps: int = 10):
    """
    Watch a trained agent play Snake.
    
    Args:
        model_path: Path to trained model
        config: Configuration
        num_games: Number of games to play
        fps: Game speed (frames per second)
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
        render_mode=True,
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
    
    print("\n" + "=" * 40)
    print("WATCHING TRAINED AGENT PLAY")
    print("=" * 40)
    print(f"Model: {model_path}")
    print(f"Grid size: {config.env.grid_size}")
    print(f"Press Q or close window to quit")
    print("=" * 40 + "\n")
    
    # Initialize pygame before the game loop
    pygame.init()
    
    total_score = 0
    games_played = 0
    
    try:
        for game in range(num_games):
            state = env.reset()
            done = False
            
            print(f"Game {game + 1}/{num_games}")
            
            while not done:
                # Check for quit events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            raise KeyboardInterrupt
                
                # Agent selects action
                action = agent.select_action(state, training=False)
                state, reward, done, info = env.step(action)
                env.render(fps=fps)
            
            print(f"  Score: {env.score}")
            total_score += env.score
            games_played += 1
            
            # Brief pause between games
            pygame.time.wait(500)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        env.close()
        
        if games_played > 0:
            print(f"\nResults:")
            print(f"  Games played: {games_played}")
            print(f"  Total score: {total_score}")
            print(f"  Average score: {total_score / games_played:.2f}")


def play_human(config: Config = None, fps: int = 10):
    """
    Play Snake manually with keyboard controls.
    
    Controls:
        Arrow keys or WASD to move
        Q to quit
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
        render_mode=True,
    )
    
    print("\n" + "=" * 40)
    print("HUMAN PLAY MODE")
    print("=" * 40)
    print("Controls:")
    print("  Arrow keys or WASD: Turn left/right")
    print("  Q: Quit")
    print("=" * 40 + "\n")
    
    # Initialize pygame before the game loop
    pygame.init()
    
    total_score = 0
    games_played = 0
    
    try:
        while True:
            env.reset()
            done = False
            action = 0  # Default: go straight
            
            print(f"Game {games_played + 1}")
            
            while not done:
                # Handle input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            raise KeyboardInterrupt
                        elif event.key in (pygame.K_LEFT, pygame.K_a):
                            action = 1  # Turn left
                        elif event.key in (pygame.K_RIGHT, pygame.K_d):
                            action = 2  # Turn right
                        else:
                            action = 0  # Go straight
                
                _, _, done, _ = env.step(action)
                env.render(fps=fps)
                action = 0  # Reset to straight
            
            print(f"  Score: {env.score}")
            total_score += env.score
            games_played += 1
            
            pygame.time.wait(500)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        env.close()
        
        if games_played > 0:
            print(f"\nResults:")
            print(f"  Games played: {games_played}")
            print(f"  Total score: {total_score}")
            print(f"  Average score: {total_score / games_played:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Watch or play Snake")
    
    parser.add_argument("model", type=str, nargs='?', default="checkpoints/best_model.pt",
                        help="Path to trained model")
    parser.add_argument("--human", action="store_true",
                        help="Play manually instead of watching AI")
    parser.add_argument("--games", type=int, default=5,
                        help="Number of games to play")
    parser.add_argument("--fps", type=int, default=10,
                        help="Game speed")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
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
    
    if args.human:
        play_human(config=config, fps=args.fps)
    else:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            print("Train a model first using: python train.py")
            print("Or play manually using: python play.py --human")
            sys.exit(1)
        
        watch_agent(args.model, config=config, num_games=args.games, fps=args.fps)


if __name__ == "__main__":
    main()

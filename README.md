# Snake Game with Reinforcement Learning

A Deep Q-Network (DQN) implementation for training an AI agent to play the classic Snake game. This project demonstrates how reinforcement learning can be applied to learn complex behaviors through trial and error.

## Project Overview

This project uses reinforcement learning to train an agent to play Snake. The game is modeled as a Markov Decision Process (MDP), where:
- **State**: Position, direction, food location, and danger detection
- **Actions**: Go straight, turn left, or turn right
- **Rewards**: Positive for eating food, negative for dying, small penalty for each step

### Features

- **Deep Q-Network (DQN)**: Neural network for Q-value approximation
- **Double DQN**: Reduces overestimation of Q-values
- **Dueling DQN**: Separates value and advantage estimation
- **Prioritized Experience Replay**: Focus on important transitions
- **Configurable hyperparameters**: Easily tune all parameters
- **Training visualization**: Track learning progress with plots
- **Interactive demo**: Watch trained agents or play manually

## Team Members

- **Spencer Conn** - Frontend & Visualization
- **Abdur Islam** - AI Development
- **Seth Aguilar** - Backend Systems
- **Quinn McCarty** - AI Development

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/seth-aguilar/cpts-437-rl-snake-game.git
cd cpts-437-rl-snake-game
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a new agent with default settings:
```bash
python train.py
```

Train with custom options:
```bash
# Train for more episodes
python train.py --episodes 2000

# Use Dueling DQN architecture
python train.py --dueling

# Use Prioritized Experience Replay
python train.py --prioritized

# Fast training mode (smaller grid, fewer episodes)
python train.py --fast

# Train with visualization
python train.py --render

# Resume training from checkpoint
python train.py --resume checkpoints/checkpoint_ep500.pt
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py checkpoints/best_model.pt

# Evaluate with visualization
python evaluate.py checkpoints/best_model.pt --render --episodes 50

# Compare multiple models
python evaluate.py --compare checkpoints/model1.pt checkpoints/model2.pt
```

### Playing

Watch the trained agent play:
```bash
python play.py

# Play with specific model
python play.py checkpoints/best_model.pt --games 10

# Play manually
python play.py --human
```

### Demo (Random Agent)

Run the original demo with random actions:
```bash
python -m demo.demo_game
```

## Project Structure

```
cpts-437-rl-snake-game/
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py      # DQN Agent implementation
│   ├── networks.py       # Neural network architectures
│   └── replay_buffer.py  # Experience replay buffer
├── env/
│   ├── __init__.py
│   └── snake_env.py      # Snake game environment
├── demo/
│   ├── __init__.py
│   └── demo_game.py      # Demo with random agent
├── utils/
│   ├── __init__.py
│   └── metrics.py        # Training metrics and visualization
├── checkpoints/          # Saved model checkpoints
├── logs/                 # Training logs and plots
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── play.py               # Interactive play script
├── config.py             # Configuration management
├── requirements.txt      # Python dependencies
└── README.md
```

## How It Works

### Markov Decision Process

The Snake game is modeled as an MDP:

1. **State Space**: The agent observes:
   - Danger detection (straight, left, right)
   - Current direction (one-hot encoded)
   - Relative position to food (normalized)

2. **Action Space**: Three relative actions:
   - `0`: Continue straight
   - `1`: Turn left
   - `2`: Turn right

3. **Reward Function**:
   - `+10.0`: Eating food
   - `-10.0`: Dying (collision)
   - `-5.0`: Getting stuck (timeout)
   - `-0.01`: Each step (encourages efficiency)

### Deep Q-Network

The DQN approximates Q-values using a neural network:

```
Q(s, a) ≈ neural_network(state)[action]
```

**Training Process**:
1. Agent interacts with environment using ε-greedy exploration
2. Transitions stored in replay buffer
3. Random mini-batches sampled for training
4. Network updated using Bellman equation:
   ```
   Q(s, a) = r + γ * max_a' Q(s', a')
   ```

### Improvements

- **Double DQN**: Uses online network to select actions, target network to evaluate
- **Dueling DQN**: Separates state value V(s) and advantage A(s,a)
- **Prioritized Replay**: Samples important transitions more frequently

## Configuration

All hyperparameters can be configured in `config.py` or via command line:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 20×20 | Game grid dimensions |
| `learning_rate` | 0.001 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Epsilon decay per episode |
| `buffer_size` | 100,000 | Replay buffer capacity |
| `batch_size` | 64 | Training batch size |
| `target_update_freq` | 100 | Target network update frequency |

## Expected Results

After training, the agent should demonstrate:
- Consistent food collection behavior
- Wall and self-collision avoidance
- Increasingly efficient movement patterns
- Scores significantly better than random policy

Typical training progress:
- Episodes 0-200: Random exploration, learning basic survival
- Episodes 200-500: Improved food collection, fewer collisions
- Episodes 500-1000: Strategic movement, higher scores
- Episodes 1000+: Fine-tuning, optimal play patterns

## Visualization

Training progress is automatically saved to `logs/` directory:
- `metrics_*.json`: Raw training data
- `training_plot_*.png`: Learning curves

The plots show:
- Score over episodes
- Survival time
- Epsilon decay
- Evaluation performance

## References

- Mnih et al., "Human-level control through deep reinforcement learning" (2015)
- Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning" (2016)
- Schaul et al., "Prioritized Experience Replay" (2015)

## License

This project is for educational purposes as part of CPTS 437 at Washington State University.
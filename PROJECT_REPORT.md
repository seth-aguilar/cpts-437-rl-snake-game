# CPTS 437: Reinforcement Learning Snake Game
## Project Report

---

## 1. Executive Summary

This project implements a **Deep Q-Network (DQN)** agent that learns to play the classic Snake game through reinforcement learning. The agent achieved a **maximum score of 65** and a **consistent average of 31+ points** on a 20×20 grid after 2,000 episodes of curriculum-based training.

### Key Results
| Metric | Value |
|--------|-------|
| Max Score | 65 |
| Final Avg Score | 31.23 |
| Max Survival | 500+ steps |
| Training Time | ~30 minutes (CUDA) |
| Grid Size | 20×20 |

---

## 2. Problem Formulation

### 2.1 The Snake Game as an MDP

We formulate Snake as a **Markov Decision Process (MDP)**:

- **State Space (S)**: 21-dimensional feature vector representing:
  - Danger detection (immediate, look-ahead, diagonal)
  - Food location relative to snake
  - Current direction
  - Path freedom metrics
  
- **Action Space (A)**: 3 discrete actions
  - `0`: Go Straight
  - `1`: Turn Left
  - `2`: Turn Right

- **Reward Function (R)**:
  ```
  R(s, a, s') = 
      +10.0   if food eaten
      -10.0   if death (collision)
      -5.0    if stuck too long
      +0.1    if moving closer to food
      -0.1    if moving away from food
      -0.01   step penalty (encourages efficiency)
  ```

- **Discount Factor (γ)**: 0.99

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Loop                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│   │              │     │              │     │              │   │
│   │  Snake Env   │────▶│  DQN Agent   │────▶│   Metrics    │   │
│   │   (State)    │◀────│   (Action)   │     │   Logger     │   │
│   │              │     │              │     │              │   │
│   └──────────────┘     └──────────────┘     └──────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                    ┌──────────────────┐                        │
│                    │  Replay Buffer   │                        │
│                    │  (50,000 trans)  │                        │
│                    └──────────────────┘                        │
│                              │                                  │
│                              ▼                                  │
│        ┌─────────────────────────────────────┐                 │
│        │         Neural Networks              │                 │
│        │  ┌─────────────┐  ┌─────────────┐   │                 │
│        │  │ Policy Net  │  │ Target Net  │   │                 │
│        │  │   (Q(s,a))  │  │  (Q'(s,a))  │   │                 │
│        │  └─────────────┘  └─────────────┘   │                 │
│        └─────────────────────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Overview

| Component | File | Purpose |
|-----------|------|---------|
| Snake Environment | `env/snake_env.py` | Game logic, state generation, rewards |
| DQN Agent | `agent/dqn_agent.py` | Action selection, learning, model management |
| Neural Networks | `agent/networks.py` | DQN, Dueling DQN architectures |
| Replay Buffer | `agent/replay_buffer.py` | Experience storage and sampling |
| Curriculum Trainer | `curriculum.py` | Progressive difficulty training |
| Configuration | `config.py` | Hyperparameters and settings |

---

## 4. Neural Network Architecture

### 4.1 Standard DQN

```
Input Layer (21 features)
        │
        ▼
┌─────────────────┐
│  Linear(21, 256)│
│     ReLU        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Linear(256, 256)│
│     ReLU        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Linear(256, 3) │
│   Q-values      │
└─────────────────┘
```

**Network Details:**
- **Input**: 21-dimensional state vector
- **Hidden Layers**: 2 fully-connected layers with 256 neurons each
- **Activation**: ReLU (Rectified Linear Unit)
- **Output**: 3 Q-values (one per action)
- **Loss**: Huber Loss (SmoothL1) for stability

### 4.2 Dueling DQN Architecture (Alternative)

```
         Input (21)
             │
             ▼
      Feature Layer (256)
             │
        ┌────┴────┐
        │         │
        ▼         ▼
   ┌─────────┐  ┌─────────┐
   │  Value  │  │Advantage│
   │ Stream  │  │ Stream  │
   │  V(s)   │  │  A(s,a) │
   └────┬────┘  └────┬────┘
        │            │
        └─────┬──────┘
              │
              ▼
        Q(s,a) = V(s) + A(s,a) - mean(A)
```

**Key Insight**: Separates state value estimation from action advantage, enabling better learning of which states are valuable regardless of action.

---

## 5. State Representation

### 5.1 Enhanced 21-Feature State Vector

Our state representation was carefully designed to provide the agent with:

```python
state = [
    # Immediate Danger (3 features) - Can I die in 1 step?
    danger_straight_1,    # Collision if go straight
    danger_left_1,        # Collision if turn left
    danger_right_1,       # Collision if turn right
    
    # Look-Ahead Danger (3 features) - Can I die in 2 steps?
    danger_straight_2,    # Danger 2 cells ahead
    danger_left_2,        # Danger 2 cells left
    danger_right_2,       # Danger 2 cells right
    
    # Diagonal Danger (2 features) - Trap detection
    danger_diag_left,     # Diagonal front-left
    danger_diag_right,    # Diagonal front-right
    
    # Path Freedom (3 features) - Open space metric
    space_straight,       # Open cells ahead (0-1)
    space_left,           # Open cells left (0-1)
    space_right,          # Open cells right (0-1)
    
    # Direction Encoding (4 features) - One-hot
    dir_up, dir_down, dir_left, dir_right,
    
    # Food Location (5 features)
    food_dx,              # Normalized x distance
    food_dy,              # Normalized y distance
    food_ahead,           # 1 if food is ahead
    food_left,            # 1 if food is to the left
    food_right,           # 1 if food is to the right
    
    # Snake Info (1 feature)
    length_ratio,         # Snake length / grid area
]
```

### 5.2 Why This Representation?

| Feature Group | Purpose | Benefit |
|---------------|---------|---------|
| Immediate Danger | Avoid death | Survival basics |
| Look-Ahead Danger | Plan 2 steps | Avoid traps |
| Diagonal Danger | Detect corners | Prevent self-trapping |
| Path Freedom | Measure open space | Favor open areas |
| Direction | Current heading | Action context |
| Food Location | Goal direction | Efficient navigation |
| Length Ratio | Snake size | Adjust behavior as snake grows |

---

## 6. Deep Q-Learning Algorithm

### 6.1 Core Algorithm

```
Algorithm: Double DQN with Experience Replay
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Initialize:
  - Policy network Q(s,a; θ)
  - Target network Q(s,a; θ⁻) ← copy of policy
  - Replay buffer D with capacity 50,000
  - ε = 1.0 (exploration rate)

For each episode:
  s ← env.reset()
  
  While not done:
    // Epsilon-greedy action selection
    if random() < ε:
      a ← random_action()
    else:
      a ← argmax_a Q(s, a; θ)
    
    // Take action, observe result
    s', r, done ← env.step(a)
    
    // Store transition
    D.store(s, a, r, s', done)
    
    // Learn from batch
    if |D| > batch_size:
      batch ← D.sample(64)
      
      // Double DQN target
      a* ← argmax_a Q(s', a; θ)        // Policy selects
      y ← r + γ · Q(s', a*; θ⁻)        // Target evaluates
      
      // Update policy network
      loss ← Huber(Q(s, a; θ), y)
      θ ← θ - α∇loss
    
    // Update target network periodically
    if steps % 100 == 0:
      θ⁻ ← θ
    
    s ← s'
  
  // Decay exploration
  ε ← max(ε_min, ε × 0.997)
```

### 6.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Algorithm** | Double DQN | Reduces Q-value overestimation |
| **Target Network** | Soft updates every 100 steps | Stabilizes training |
| **Replay Buffer** | 50,000 transitions | Balance memory/diversity |
| **Batch Size** | 64 | Standard, efficient GPU usage |
| **Loss Function** | Huber Loss | Robust to outliers |
| **Optimizer** | Adam (lr=0.001) | Adaptive learning rate |

### 6.3 Double DQN vs Standard DQN

**Standard DQN Target:**
$$y = r + \gamma \max_{a'} Q(s', a'; \theta^{-})$$

**Double DQN Target:**
$$y = r + \gamma Q(s', \underset{a'}{\text{argmax}} Q(s', a'; \theta); \theta^{-})$$

**Why Double DQN?**
- Standard DQN uses the same network to select AND evaluate actions
- This leads to systematic overestimation of Q-values
- Double DQN decouples selection (policy net) from evaluation (target net)
- Results in more stable, accurate value estimates

---

## 7. Curriculum Learning

### 7.1 Progressive Training Strategy

Instead of training directly on the full 20×20 grid, we use curriculum learning:

```
┌─────────────────────────────────────────────────────────────────┐
│              CURRICULUM LEARNING PROGRESSION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Stage 1        Stage 2        Stage 3        Stage 4          │
│  ┌──────┐      ┌────────┐    ┌──────────┐   ┌────────────┐     │
│  │ 8×8  │  ──▶ │  12×12 │ ──▶│  16×16   │──▶│   20×20    │     │
│  │ Tiny │      │ Small  │    │  Medium  │   │   Full     │     │
│  └──────┘      └────────┘    └──────────┘   └────────────┘     │
│                                                                 │
│  Thresh: 2.5   Thresh: 4.0   Thresh: 6.0   Thresh: 12.0        │
│                                                                 │
│  Episodes:150  Episodes:200  Episodes:300  Episodes:1400       │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Stage Configuration

| Stage | Grid Size | Timeout | Promotion Threshold |
|-------|-----------|---------|---------------------|
| Tiny | 8×8 | 100 steps | Avg score ≥ 2.5 |
| Small | 12×12 | 150 steps | Avg score ≥ 4.0 |
| Medium | 16×16 | 200 steps | Avg score ≥ 6.0 |
| Full | 20×20 | 200 steps | Final training |

### 7.3 Benefits of Curriculum Learning

1. **Faster Initial Learning**: Smaller grid = less state space = faster convergence
2. **Transfer Learning**: Skills transfer as difficulty increases
3. **Exploration Efficiency**: High epsilon useful in small grids
4. **Stable Training**: Gradual difficulty prevents catastrophic failures

---

## 8. Training Results

### 8.1 Training Progression

From the training log (2000 episodes):

```
Stage 1 (8×8):   381 episodes  →  Avg Score: 2.54 ✓
Stage 2 (12×12):  50 episodes  →  Avg Score: 4.02 ✓
Stage 3 (16×16): 169 episodes  →  Avg Score: 6.06 ✓
Stage 4 (20×20): 1400 episodes →  Avg Score: 31.23 ✓
```

### 8.2 Performance Metrics

| Metric | Early Training | Final Performance |
|--------|----------------|-------------------|
| Avg Score | 0.10 | 31.23 |
| Max Score | 1 | 65 |
| Avg Survival | ~20 steps | 493.9 steps |
| Epsilon | 1.0 | 0.01 |

### 8.3 Learning Curve Insights

1. **Exploration Phase** (Episodes 1-300): Random behavior, learning basics
2. **Rapid Improvement** (Episodes 300-800): Q-values stabilize, scores climb
3. **Refinement** (Episodes 800-2000): Fine-tuning, consistent high performance

---

## 9. Hyperparameter Summary

### 9.1 Final Configuration

```python
# Agent Hyperparameters
hidden_dims = [256, 256]      # Neural network architecture
learning_rate = 0.001         # Adam optimizer LR
gamma = 0.99                  # Discount factor
epsilon_start = 1.0           # Initial exploration
epsilon_end = 0.01            # Minimum exploration
epsilon_decay = 0.997         # Per-episode decay
buffer_size = 50000           # Replay buffer capacity
batch_size = 64               # Training batch size
target_update_freq = 100      # Target network sync

# Environment
grid_size = (20, 20)          # Game board
food_reward = 10.0            # +10 for eating
death_penalty = -10.0         # -10 for dying
step_penalty = -0.01          # Small time pressure
reward_shaping = True         # Distance-based rewards
```

### 9.2 Hyperparameter Rationale

| Parameter | Value | Why |
|-----------|-------|-----|
| γ = 0.99 | High | Long-term planning for food collection |
| ε decay = 0.997 | Fast | Curriculum already provides structured exploration |
| Buffer = 50k | Medium | Balance freshness and diversity |
| LR = 0.001 | Standard | Stable with Adam |
| Hidden = [256,256] | Deep enough | Sufficient for 21-dim input |

---

## 10. Key Architectural Decisions

### 10.1 Why Relative Actions (Not Absolute)?

**Relative**: `[Straight, Left, Right]` (3 actions)

**Absolute**: `[Up, Down, Left, Right]` (4 actions)

**Decision**: We chose **relative actions** because:
- Snake cannot reverse direction (would die instantly)
- Reduces invalid action space
- State representation is relative to heading
- Simpler learning problem

### 10.2 Why Feature Vector (Not CNN)?

**Feature Vector**: 21 hand-crafted features

**CNN**: Raw grid pixels

**Decision**: We chose **feature vector** because:
- Incorporates domain knowledge (danger, food direction)
- Faster training (smaller input)
- More interpretable
- Sufficient for this problem complexity
- CNN would require much more data/time

### 10.3 Why Double DQN?

Standard DQN has a known **overestimation bias** due to:
$$\max_a Q(s,a) \geq Q(s, \text{optimal action})$$

Double DQN fixes this by:
1. Using policy network to **select** best action
2. Using target network to **evaluate** that action

Result: More accurate Q-values, stable training.

### 10.4 Why Curriculum Learning?

Direct training on 20×20 grid is challenging because:
- Large state space
- Sparse rewards (food far from spawn)
- High exploration needed

Curriculum learning provides:
- Denser rewards in small grids
- Gradual complexity increase
- Transferred knowledge between stages

---

## 11. Project Structure

```
cpts-437-rl-snake-game/
├── agent/
│   ├── __init__.py
│   ├── dqn_agent.py        # DQN agent implementation
│   ├── networks.py         # Neural network architectures
│   └── replay_buffer.py    # Experience replay
├── env/
│   ├── __init__.py
│   └── snake_env.py        # Snake game environment
├── utils/
│   ├── __init__.py
│   └── metrics.py          # Logging and visualization
├── config.py               # Configuration management
├── train.py                # Standard training script
├── curriculum.py           # Curriculum learning trainer
├── evaluate.py             # Model evaluation
├── play.py                 # Watch agent play
├── visualize.py            # Generate training graphs
├── checkpoints/            # Saved models
├── logs/                   # Training logs
└── visualizations/         # Generated graphs
```

---

## 12. Usage Guide

### 12.1 Training

```bash
# Standard training
python train.py --episodes 1000

# Curriculum learning (recommended)
python curriculum.py --episodes 2000

# With custom settings
python train.py --episodes 2000 --double-dqn --lr 0.0005
```

### 12.2 Evaluation

```bash
# Watch trained agent
python play.py checkpoints/curriculum_model.pt --games 5

# Benchmark evaluation
python evaluate.py checkpoints/curriculum_model.pt --episodes 100
```

### 12.3 Visualization

```bash
# Generate all graphs
python visualize.py

# From specific metrics file
python visualize.py logs/metrics_20251201_171641.json
```

---

## 13. Future Improvements

1. **Prioritized Experience Replay**: Sample important transitions more often
2. **Dueling DQN**: Enable for better state-action separation
3. **Noisy Networks**: Replace ε-greedy with learned exploration
4. **Rainbow DQN**: Combine multiple improvements
5. **CNN Alternative**: For pixel-based learning comparison
6. **Multi-Agent**: Train multiple snakes simultaneously

---

## 14. Conclusion

This project successfully demonstrates the application of Deep Reinforcement Learning to the Snake game. Key achievements:

✅ **Effective State Design**: 21-feature representation enabling trap avoidance

✅ **Stable Training**: Double DQN with experience replay

✅ **Progressive Learning**: Curriculum approach accelerating convergence

✅ **Strong Performance**: 65 max score, 31+ average on 20×20 grid

The agent learned sophisticated behaviors including food-seeking, wall avoidance, and importantly, avoiding self-trapping through the enhanced look-ahead and open-space features.

---

## References

1. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
2. Van Hasselt, H., et al. (2016). *Deep Reinforcement Learning with Double Q-learning*. AAAI.
3. Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning*. ICML.
4. Bengio, Y., et al. (2009). *Curriculum Learning*. ICML.

---

*CPTS 437 - Introduction to Reinforcement Learning*
*Washington State University*
*December 2025*

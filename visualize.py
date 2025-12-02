"""
Advanced visualization for RL Snake training metrics.

Creates publication-quality graphs from training data including:
- Learning curves with confidence bands
- Performance distributions
- Curriculum stage progression
- Score heatmaps and histograms
"""

import os
import json
import argparse
import numpy as np
from typing import List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    exit(1)


def load_metrics(filepath: str) -> dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def moving_average(data: List[float], window: int) -> np.ndarray:
    """Calculate moving average."""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def moving_std(data: List[float], window: int) -> np.ndarray:
    """Calculate moving standard deviation."""
    data = np.array(data)
    stds = []
    for i in range(window - 1, len(data)):
        stds.append(np.std(data[i - window + 1:i + 1]))
    return np.array(stds)


def plot_learning_curve_with_bands(ax, scores: List[int], window: int = 50, 
                                    color: str = '#2E86AB', label: str = 'Score'):
    """Plot learning curve with confidence bands."""
    episodes = np.arange(1, len(scores) + 1)
    
    # Raw scores (faded)
    ax.plot(episodes, scores, alpha=0.15, color=color, linewidth=0.5)
    
    # Moving average
    ma = moving_average(scores, window)
    ma_episodes = np.arange(window, len(scores) + 1)
    ax.plot(ma_episodes, ma, color=color, linewidth=2.5, label=f'{label} (MA-{window})')
    
    # Confidence band (±1 std)
    std = moving_std(scores, window)
    ax.fill_between(ma_episodes, ma - std, ma + std, alpha=0.2, color=color)
    
    return ma[-1] if len(ma) > 0 else 0


def create_comprehensive_visualization(metrics_path: str, save_dir: str = "visualizations"):
    """Create a comprehensive multi-panel visualization."""
    
    os.makedirs(save_dir, exist_ok=True)
    data = load_metrics(metrics_path)
    
    scores = data['episode_scores']
    lengths = data['episode_lengths']
    epsilons = data['epsilons']
    rewards = data.get('episode_rewards', [])
    snake_lengths = data.get('episode_snake_lengths', [])
    run_id = data.get('run_id', 'unknown')
    
    n_episodes = len(scores)
    window = min(50, n_episodes // 20) if n_episodes > 100 else 10
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'RL Snake Training Analysis\nRun: {run_id} | Episodes: {n_episodes}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ===== 1. Main Learning Curve (large) =====
    ax1 = fig.add_subplot(gs[0, :2])
    final_score = plot_learning_curve_with_bands(ax1, scores, window, '#2E86AB', 'Score')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Score (Food Eaten)', fontsize=11)
    ax1.set_title('Learning Curve with Confidence Band', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    
    # Add max score annotation
    max_score = max(scores)
    max_idx = scores.index(max_score)
    ax1.annotate(f'Max: {max_score}', xy=(max_idx, max_score), 
                 xytext=(max_idx + n_episodes*0.05, max_score),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red', fontweight='bold')
    
    # ===== 2. Score Distribution =====
    ax2 = fig.add_subplot(gs[0, 2])
    
    # Split into early vs late training
    mid = len(scores) // 2
    early_scores = scores[:mid]
    late_scores = scores[mid:]
    
    bins = np.linspace(0, max(scores) + 1, 30)
    ax2.hist(early_scores, bins=bins, alpha=0.5, color='#E74C3C', label='First half', density=True)
    ax2.hist(late_scores, bins=bins, alpha=0.5, color='#27AE60', label='Second half', density=True)
    ax2.axvline(np.mean(early_scores), color='#E74C3C', linestyle='--', linewidth=2)
    ax2.axvline(np.mean(late_scores), color='#27AE60', linestyle='--', linewidth=2)
    ax2.set_xlabel('Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Score Distribution: Early vs Late', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    
    # ===== 3. Survival Time =====
    ax3 = fig.add_subplot(gs[1, 0])
    plot_learning_curve_with_bands(ax3, lengths, window, '#8E44AD', 'Steps')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('Steps Survived', fontsize=11)
    ax3.set_title('Survival Time', fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    
    # ===== 4. Epsilon Decay =====
    ax4 = fig.add_subplot(gs[1, 1])
    episodes = np.arange(1, len(epsilons) + 1)
    ax4.plot(episodes, epsilons, color='#F39C12', linewidth=2)
    ax4.fill_between(episodes, 0, epsilons, alpha=0.3, color='#F39C12')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Epsilon', fontsize=11)
    ax4.set_title('Exploration Rate Decay', fontsize=13, fontweight='bold')
    ax4.set_ylim(0, 1.05)
    
    # Mark key epsilon values
    for target_eps in [0.5, 0.1, 0.01]:
        if min(epsilons) < target_eps:
            idx = next(i for i, e in enumerate(epsilons) if e <= target_eps)
            ax4.axhline(target_eps, color='gray', linestyle=':', alpha=0.5)
            ax4.scatter([idx], [target_eps], color='#E74C3C', s=50, zorder=5)
            ax4.annotate(f'ε={target_eps} @ ep {idx}', xy=(idx, target_eps),
                        xytext=(idx + n_episodes*0.1, target_eps + 0.05),
                        fontsize=8, alpha=0.8)
    
    # ===== 5. Score vs Survival Correlation =====
    ax5 = fig.add_subplot(gs[1, 2])
    scatter = ax5.scatter(lengths, scores, alpha=0.3, c=range(len(scores)), 
                         cmap='viridis', s=10)
    ax5.set_xlabel('Episode Length (Steps)', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_title('Score vs Survival Time', fontsize=13, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(lengths, scores, 1)
    p = np.poly1d(z)
    ax5.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8, linewidth=2, label='Trend')
    ax5.legend(fontsize=9)
    
    cbar = plt.colorbar(scatter, ax=ax5, shrink=0.8)
    cbar.set_label('Episode', fontsize=9)
    
    # ===== 6. Performance Heatmap (episodes x score ranges) =====
    ax6 = fig.add_subplot(gs[2, 0])
    
    # Create bins of episodes and count score distributions
    n_bins = 10
    bin_size = n_episodes // n_bins
    score_bins = np.linspace(0, max(scores) + 1, 15)
    
    heatmap_data = []
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n_episodes
        bin_scores = scores[start:end]
        hist, _ = np.histogram(bin_scores, bins=score_bins, density=True)
        heatmap_data.append(hist)
    
    heatmap_data = np.array(heatmap_data).T
    im = ax6.imshow(heatmap_data, aspect='auto', origin='lower', cmap='YlOrRd',
                    extent=[0, n_bins, 0, max(scores)])
    ax6.set_xlabel('Training Phase (decile)', fontsize=11)
    ax6.set_ylabel('Score', fontsize=11)
    ax6.set_title('Score Distribution Over Training', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Density')
    
    # ===== 7. Rolling Statistics =====
    ax7 = fig.add_subplot(gs[2, 1])
    
    if len(scores) > window:
        ma = moving_average(scores, window)
        ma_eps = np.arange(window, len(scores) + 1)
        
        # Calculate percentiles
        p25 = []
        p75 = []
        for i in range(window - 1, len(scores)):
            window_data = scores[i - window + 1:i + 1]
            p25.append(np.percentile(window_data, 25))
            p75.append(np.percentile(window_data, 75))
        
        ax7.fill_between(ma_eps, p25, p75, alpha=0.3, color='#3498DB', label='IQR')
        ax7.plot(ma_eps, ma, color='#2C3E50', linewidth=2, label='Mean')
        ax7.set_xlabel('Episode', fontsize=11)
        ax7.set_ylabel('Score', fontsize=11)
        ax7.set_title(f'Rolling Statistics (window={window})', fontsize=13, fontweight='bold')
        ax7.legend(fontsize=9)
    
    # ===== 8. Summary Stats Box =====
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary = data.get('summary', {})
    stats_text = f"""
    ╔══════════════════════════════════╗
    ║       TRAINING SUMMARY           ║
    ╠══════════════════════════════════╣
    ║  Total Episodes:    {n_episodes:>10}   ║
    ║  Max Score:         {max(scores):>10}   ║
    ║  Final Avg Score:   {final_score:>10.2f}   ║
    ║  Overall Avg:       {np.mean(scores):>10.2f}   ║
    ║  Std Dev:           {np.std(scores):>10.2f}   ║
    ║  Max Survival:      {max(lengths):>10}   ║
    ║  Avg Survival:      {np.mean(lengths):>10.1f}   ║
    ║  Final Epsilon:     {epsilons[-1]:>10.4f}   ║
    ╚══════════════════════════════════╝
    
    Improvement: {np.mean(late_scores) - np.mean(early_scores):+.2f} score
    ({(np.mean(late_scores)/np.mean(early_scores) - 1)*100 if np.mean(early_scores) > 0 else 0:+.1f}% increase)
    """
    
    ax8.text(0.1, 0.5, stats_text, transform=ax8.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#ECF0F1', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save figure
    save_path = os.path.join(save_dir, f"analysis_{run_id}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Comprehensive analysis saved to: {save_path}")
    
    plt.show()
    return save_path


def create_milestone_chart(metrics_path: str, save_dir: str = "visualizations"):
    """Create a milestone achievement chart."""
    
    os.makedirs(save_dir, exist_ok=True)
    data = load_metrics(metrics_path)
    
    scores = data['episode_scores']
    run_id = data.get('run_id', 'unknown')
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    episodes = np.arange(1, len(scores) + 1)
    
    # Define milestones
    milestones = [5, 10, 15, 20, 25, 30, 40, 50, 60]
    colors = plt.cm.viridis(np.linspace(0, 1, len(milestones)))
    
    # Plot base scores
    ax.plot(episodes, scores, alpha=0.3, color='gray', linewidth=0.5)
    ma = moving_average(scores, 50)
    ax.plot(range(50, len(scores) + 1), ma, color='#2E86AB', linewidth=2.5, label='Moving Avg')
    
    # Find when each milestone was first achieved
    achieved_milestones = []
    for i, milestone in enumerate(milestones):
        first_episode = next((ep for ep, score in enumerate(scores, 1) if score >= milestone), None)
        if first_episode:
            achieved_milestones.append((milestone, first_episode))
            ax.axhline(milestone, color=colors[i], linestyle='--', alpha=0.4, linewidth=1)
            ax.scatter([first_episode], [milestone], color=colors[i], s=100, zorder=5,
                      edgecolors='black', linewidth=1)
            ax.annotate(f'Score {milestone}\n(ep {first_episode})', 
                       xy=(first_episode, milestone),
                       xytext=(first_episode + len(scores)*0.03, milestone + 2),
                       fontsize=9, fontweight='bold', color=colors[i])
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Milestone Achievement Timeline\nRun: {run_id}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim(0, max(scores) * 1.1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"milestones_{run_id}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Milestone chart saved to: {save_path}")
    
    plt.show()
    return save_path


def create_performance_dashboard(metrics_path: str, save_dir: str = "visualizations"):
    """Create a simple performance dashboard."""
    
    os.makedirs(save_dir, exist_ok=True)
    data = load_metrics(metrics_path)
    
    scores = data['episode_scores']
    lengths = data['episode_lengths']
    run_id = data.get('run_id', 'unknown')
    
    n = len(scores)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Performance Dashboard - {run_id}', fontsize=14, fontweight='bold')
    
    # 1. Score over time with phases
    ax = axes[0, 0]
    phases = ['Exploration', 'Early Learning', 'Rapid Improvement', 'Fine-tuning']
    phase_colors = ['#E74C3C', '#F39C12', '#27AE60', '#3498DB']
    phase_bounds = [0, n//4, n//2, 3*n//4, n]
    
    for i, (phase, color) in enumerate(zip(phases, phase_colors)):
        start, end = phase_bounds[i], phase_bounds[i+1]
        ax.axvspan(start, end, alpha=0.15, color=color, label=phase)
        ax.plot(range(start, end), scores[start:end], alpha=0.5, color=color, linewidth=0.8)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_title('Training Phases')
    ax.legend(loc='upper left', fontsize=8)
    
    # 2. Cumulative score
    ax = axes[0, 1]
    cumulative = np.cumsum(scores)
    ax.fill_between(range(n), cumulative, alpha=0.5, color='#9B59B6')
    ax.plot(cumulative, color='#8E44AD', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Score')
    ax.set_title('Total Food Collected Over Training')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 3. Efficiency (score per step)
    ax = axes[1, 0]
    efficiency = [s/l if l > 0 else 0 for s, l in zip(scores, lengths)]
    window = 50
    eff_ma = moving_average(efficiency, window)
    ax.plot(range(window, n + 1), eff_ma, color='#16A085', linewidth=2)
    ax.fill_between(range(window, n + 1), eff_ma, alpha=0.3, color='#1ABC9C')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score / Steps')
    ax.set_title('Learning Efficiency (Score per Step)')
    
    # 4. Top 10 scores
    ax = axes[1, 1]
    top_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:10]
    top_episodes = [t[0] + 1 for t in top_scores]
    top_values = [t[1] for t in top_scores]
    
    bars = ax.barh(range(10), top_values, color=plt.cm.RdYlGn(np.linspace(0.8, 0.2, 10)))
    ax.set_yticks(range(10))
    ax.set_yticklabels([f'Ep {ep}' for ep in top_episodes])
    ax.set_xlabel('Score')
    ax.set_title('Top 10 Episodes')
    ax.invert_yaxis()
    
    for i, (bar, val) in enumerate(zip(bars, top_values)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val}',
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"dashboard_{run_id}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"✓ Performance dashboard saved to: {save_path}")
    
    plt.show()
    return save_path


def main():
    parser = argparse.ArgumentParser(description='Visualize RL Snake training metrics')
    parser.add_argument('metrics_file', nargs='?', help='Path to metrics JSON file')
    parser.add_argument('--all', '-a', action='store_true', 
                       help='Generate all visualizations')
    parser.add_argument('--comprehensive', '-c', action='store_true',
                       help='Generate comprehensive analysis')
    parser.add_argument('--milestones', '-m', action='store_true',
                       help='Generate milestone chart')
    parser.add_argument('--dashboard', '-d', action='store_true',
                       help='Generate performance dashboard')
    parser.add_argument('--output-dir', '-o', default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Find metrics file
    if args.metrics_file:
        metrics_path = args.metrics_file
    else:
        # Find most recent metrics file
        logs_dir = 'logs'
        if os.path.exists(logs_dir):
            json_files = [f for f in os.listdir(logs_dir) if f.endswith('.json')]
            if json_files:
                json_files.sort(reverse=True)
                metrics_path = os.path.join(logs_dir, json_files[0])
                print(f"Using most recent metrics: {metrics_path}")
            else:
                print("No metrics files found in logs/")
                return
        else:
            print("No logs directory found")
            return
    
    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}")
        return
    
    print(f"\n{'='*60}")
    print(f"RL Snake Training Visualizer")
    print(f"{'='*60}")
    print(f"Loading: {metrics_path}\n")
    
    # Default to all if no specific option selected
    if not (args.comprehensive or args.milestones or args.dashboard):
        args.all = True
    
    if args.all or args.comprehensive:
        create_comprehensive_visualization(metrics_path, args.output_dir)
    
    if args.all or args.milestones:
        create_milestone_chart(metrics_path, args.output_dir)
    
    if args.all or args.dashboard:
        create_performance_dashboard(metrics_path, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"All visualizations saved to: {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

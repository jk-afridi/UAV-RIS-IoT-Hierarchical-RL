#!/usr/bin/env python3
"""
=============================================================================
PUBLICATION FIGURE GENERATOR — 4-Way Comparison
=============================================================================
Generates IEEE Transactions-quality figures comparing:
  1. Proposed Hierarchical DRL (PPO + SAC, β_min = 0.2)
  2. Ideal RIS Upper Bound (PPO + SAC, β_min = 1.0)
  3. PPO-Only Baseline (trajectory learned, RIS random)
  4. Random Baseline (everything random)

Output: PNG (300 DPI) + PDF (vector) for LaTeX \includegraphics
=============================================================================
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

# ============================================================================
# IEEE STYLE CONFIGURATION
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.8,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

# ============================================================================
# PROFESSIONAL COLOR PALETTE (colorblind-safe, publication-grade)
# ============================================================================
COLORS = {
    'proposed':  '#1B4F72',   # Deep navy blue
    'ideal':     '#1E8449',   # Rich emerald green
    'ppo_only':  '#D4AC0D',   # Gold/amber
    'random':    '#C0392B',   # Rich crimson red
}

FILL_ALPHA = {
    'proposed': 0.15,
    'ideal':    0.12,
    'ppo_only': 0.10,
    'random':   0.08,
}

MARKERS = {
    'proposed': 'o',
    'ideal':    's',
    'ppo_only': '^',
    'random':   'D',
}

LINESTYLES = {
    'proposed': '-',
    'ideal':    '--',
    'ppo_only': '-.',
    'random':   ':',
}

LABELS = {
    'proposed': r'Proposed (PPO+SAC, $\beta_{min}$=0.2)',
    'ideal':    r'Ideal RIS ($\beta_{min}$=1.0, Upper Bound)',
    'ppo_only': 'PPO-Only (No RIS Optimization)',
    'random':   'Random Baseline',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ema_smooth(data, alpha=0.02):
    """Exponential Moving Average for smooth convergence curves."""
    smoothed = np.zeros_like(data, dtype=float)
    smoothed[0] = data[0]
    for i in range(1, len(data)):
        smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed


def rolling_std(data, window=200):
    """Rolling standard deviation for confidence bands."""
    stds = np.zeros_like(data, dtype=float)
    for i in range(len(data)):
        start = max(0, i - window)
        stds[i] = np.std(data[start:i + 1])
    return stds


def add_marker_sparse(ax, x, y, color, marker, interval=500):
    """Add sparse markers along a curve for print clarity."""
    idx = np.arange(0, len(x), interval)
    ax.scatter(x[idx], y[idx], color=color, marker=marker,
               s=30, zorder=5, edgecolors='white', linewidths=0.5)


# ============================================================================
# LOAD DATA
# ============================================================================

RESULTS_BASE = Path(__file__).resolve().parents[2] / 'results'

def load_npz(path):
    data = np.load(path)
    return {k: data[k] for k in data.files}

proposed = load_npz(RESULTS_BASE / 'hierarchical' / 'beta0.2_rmin1 (2).25_seed42_colab' / 'training_metrics.npz')
ideal    = load_npz(RESULTS_BASE / 'hierarchical' / 'beta1.0_rmin1.25_seed42' / 'training_metrics.npz')
ppo_only = load_npz(RESULTS_BASE / 'ppo_only' / 'beta0.2_rmin1.25_seed42' / 'training_metrics.npz')
random_b = load_npz(RESULTS_BASE / 'random' / 'beta0.2_rmin1.25_seed42' / 'training_metrics_random.npz')
# Legacy random-baseline files store decode_rates as a 0-1 FRACTION while every other
# method stores PERCENT; plotting them together understated the random baseline 100x
# (0.03% shown instead of 2.76%). train_random_baseline.py now stores percent; this
# rescales files produced by the old version.
if random_b['decode_rates'].max() <= 1.0:
    random_b['decode_rates'] = random_b['decode_rates'] * 100.0

ALL = {
    'proposed': proposed,
    'ideal':    ideal,
    'ppo_only': ppo_only,
    'random':   random_b,
}

# Output directory
FIG_DIR = RESULTS_BASE / 'figures_4way'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    print(f'[OK] {name} saved → {FIG_DIR / name}.png')


# ============================================================================
# FIGURE 1: Average AoI Convergence (4 curves)
# ============================================================================

def fig1_aoi_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in ['ideal', 'proposed', 'ppo_only', 'random']:
        data = ALL[key]['avg_aois']
        episodes = np.arange(len(data))
        smoothed = ema_smooth(data, alpha=0.02)
        std = rolling_std(data, window=200) * 0.5

        ax.plot(episodes, smoothed, color=COLORS[key],
                linestyle=LINESTYLES[key], label=LABELS[key], zorder=3)
        ax.fill_between(episodes, smoothed - std, smoothed + std,
                        color=COLORS[key], alpha=FILL_ALPHA[key], zorder=2)
        add_marker_sparse(ax, episodes, smoothed, COLORS[key], MARKERS[key])

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Average Age of Information (AoI)')
    ax.set_title('Average AoI Convergence Across Methods')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray',
              fancybox=True, shadow=False)
    ax.set_xlim(0, 5000)
    ax.set_ylim(19, 28)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Add annotation arrow pointing to proposed convergence
    ax.annotate('Proposed converges\nto AoI ≈ 22.8',
                xy=(4000, 22.8), xytext=(3000, 19.8),
                fontsize=8.5, color=COLORS['proposed'],
                arrowprops=dict(arrowstyle='->', color=COLORS['proposed'],
                                lw=1.2, connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=COLORS['proposed'], alpha=0.9))

    fig.tight_layout()
    save_fig(fig, 'fig1_avg_aoi_4way')
    plt.close(fig)


# ============================================================================
# FIGURE 2: Decode Rate Convergence (4 curves)
# ============================================================================

def fig2_decode_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in ['ideal', 'proposed', 'ppo_only', 'random']:
        data = ALL[key]['decode_rates']
        episodes = np.arange(len(data))
        smoothed = ema_smooth(data, alpha=0.02)
        std = rolling_std(data, window=200) * 0.5

        ax.plot(episodes, smoothed, color=COLORS[key],
                linestyle=LINESTYLES[key], label=LABELS[key], zorder=3)
        ax.fill_between(episodes, smoothed - std, smoothed + std,
                        color=COLORS[key], alpha=FILL_ALPHA[key], zorder=2)
        add_marker_sparse(ax, episodes, smoothed, COLORS[key], MARKERS[key])

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Successful Decode Rate (%)')
    ax.set_title('Decode Rate Convergence Across Methods')
    ax.legend(loc='center right', framealpha=0.9, edgecolor='gray',
              fancybox=True, shadow=False)
    ax.set_xlim(0, 5000)
    ax.set_ylim(-1, 25)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # Horizontal threshold line
    ax.axhline(y=15.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(200, 15.5, '15% threshold', fontsize=8, color='gray', alpha=0.7)

    # Annotation for the massive gap
    ax.annotate('',
                xy=(4500, 15.0), xytext=(4500, 4.0),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5))
    ax.text(4550, 9.5, '3.7×\ngain', fontsize=9, fontweight='bold',
            color='#555555', ha='left', va='center')

    fig.tight_layout()
    save_fig(fig, 'fig2_decode_rate_4way')
    plt.close(fig)


# ============================================================================
# FIGURE 3: Combined Dual-Axis (AoI + Decode Rate)
# ============================================================================

def fig3_dual_axis():
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Plot AoI on left axis (solid lines)
    for key in ['ideal', 'proposed', 'ppo_only', 'random']:
        data = ALL[key]['avg_aois']
        episodes = np.arange(len(data))
        smoothed = ema_smooth(data, alpha=0.015)
        ax1.plot(episodes, smoothed, color=COLORS[key],
                 linestyle='-', linewidth=1.5, alpha=0.85, zorder=3)

    # Plot Decode on right axis (dashed lines)
    for key in ['ideal', 'proposed', 'ppo_only', 'random']:
        data = ALL[key]['decode_rates']
        episodes = np.arange(len(data))
        smoothed = ema_smooth(data, alpha=0.015)
        ax2.plot(episodes, smoothed, color=COLORS[key],
                 linestyle='--', linewidth=1.5, alpha=0.85, zorder=3)

    ax1.set_xlabel('Training Episode')
    ax1.set_ylabel('Average AoI (solid lines)', color='#333333')
    ax2.set_ylabel('Decode Rate % (dashed lines)', color='#333333')

    ax1.set_xlim(0, 5000)
    ax1.set_ylim(19, 28)
    ax2.set_ylim(-1, 25)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    ax1.set_title('Training Dynamics: AoI (↓ better) vs Decode Rate (↑ better)')

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['proposed'], lw=2, label=LABELS['proposed']),
        Line2D([0], [0], color=COLORS['ideal'], lw=2, label=LABELS['ideal']),
        Line2D([0], [0], color=COLORS['ppo_only'], lw=2, label=LABELS['ppo_only']),
        Line2D([0], [0], color=COLORS['random'], lw=2, label=LABELS['random']),
        Line2D([0], [0], color='gray', linestyle='-', lw=1.5, label='AoI (solid)'),
        Line2D([0], [0], color='gray', linestyle='--', lw=1.5, label='Decode Rate (dashed)'),
    ]
    ax1.legend(handles=legend_elements, loc='center right', fontsize=8,
               framealpha=0.9, edgecolor='gray', fancybox=True)

    fig.tight_layout()
    save_fig(fig, 'fig3_dual_axis_4way')
    plt.close(fig)


# ============================================================================
# FIGURE 4: Performance Summary Bar Chart
# ============================================================================

def fig4_summary_bars():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    methods = ['Proposed\n(PPO+SAC)', 'Ideal RIS\n(β=1.0)', 'PPO-Only\n(No SAC)', 'Random']
    color_list = [COLORS['proposed'], COLORS['ideal'], COLORS['ppo_only'], COLORS['random']]
    hatch_list = ['', '///', '...', 'xxx']

    # Converged stats (last 500 episodes)
    def converged_stats(data, window=500):
        n = len(data['avg_aois'])
        w = min(window, n)
        return {
            'aoi_mean': np.mean(data['avg_aois'][-w:]),
            'aoi_std':  np.std(data['avg_aois'][-w:]),
            'dec_mean': np.mean(data['decode_rates'][-w:]),
            'dec_std':  np.std(data['decode_rates'][-w:]),
        }

    stats = [
        converged_stats(ALL['proposed']),
        converged_stats(ALL['ideal']),
        converged_stats(ALL['ppo_only']),
        converged_stats(ALL['random']),
    ]

    x = np.arange(len(methods))
    width = 0.6

    # ---- Left: Average AoI (lower is better) ----
    aoi_means = [s['aoi_mean'] for s in stats]
    aoi_stds  = [s['aoi_std'] for s in stats]

    bars1 = ax1.bar(x, aoi_means, width, yerr=aoi_stds, capsize=5,
                    color=color_list, edgecolor='white', linewidth=1.2,
                    alpha=0.88, hatch=hatch_list,
                    error_kw={'linewidth': 1.2, 'capthick': 1.2})

    ax1.set_ylabel('Average AoI (lower is better)')
    ax1.set_title('(a) Average Age of Information')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=9)
    ax1.set_ylim(18, 28)
    ax1.axhline(y=min(aoi_means), color='gray', linestyle=':', linewidth=0.7, alpha=0.4)

    # Value labels on bars
    for bar, val, std in zip(bars1, aoi_means, aoi_stds):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.15,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Best indicator arrow
    best_idx = np.argmin(aoi_means)
    ax1.annotate('Best', xy=(best_idx, aoi_means[best_idx] - 0.3),
                 fontsize=9, fontweight='bold', ha='center',
                 color=color_list[best_idx])

    # ---- Right: Decode Rate (higher is better) ----
    dec_means = [s['dec_mean'] for s in stats]
    dec_stds  = [s['dec_std'] for s in stats]

    bars2 = ax2.bar(x, dec_means, width, yerr=dec_stds, capsize=5,
                    color=color_list, edgecolor='white', linewidth=1.2,
                    alpha=0.88, hatch=hatch_list,
                    error_kw={'linewidth': 1.2, 'capthick': 1.2})

    ax2.set_ylabel('Decode Rate % (higher is better)')
    ax2.set_title('(b) Successful Decode Rate')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, fontsize=9)
    ax2.set_ylim(0, 25)

    # Value labels on bars
    for bar, val, std in zip(bars2, dec_means, dec_stds):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Improvement annotation
    ax2.annotate(f'3.7× gain\nvs PPO-Only',
                 xy=(0, dec_means[0] + dec_stds[0] + 1.5),
                 xytext=(1.5, 22),
                 fontsize=8, fontweight='bold', color=COLORS['proposed'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['proposed'], lw=1.2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F6F3',
                           edgecolor=COLORS['proposed'], alpha=0.9))

    fig.suptitle('Performance Comparison: Converged Metrics (Last 500 Episodes)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, 'fig4_summary_bars_4way')
    plt.close(fig)


# ============================================================================
# FIGURE 5: Reward Convergence (4 curves)
# ============================================================================

def fig5_reward_convergence():
    fig, ax = plt.subplots(figsize=(8, 5))

    for key in ['ideal', 'proposed', 'ppo_only', 'random']:
        data = ALL[key]['rewards']
        episodes = np.arange(len(data))
        smoothed = ema_smooth(data, alpha=0.02)
        std = rolling_std(data, window=200) * 0.4

        ax.plot(episodes, smoothed, color=COLORS[key],
                linestyle=LINESTYLES[key], label=LABELS[key], zorder=3)
        ax.fill_between(episodes, smoothed - std, smoothed + std,
                        color=COLORS[key], alpha=FILL_ALPHA[key], zorder=2)
        add_marker_sparse(ax, episodes, smoothed, COLORS[key], MARKERS[key])

    ax.set_xlabel('Training Episode')
    ax.set_ylabel('Cumulative Episode Reward')
    ax.set_title('Reward Convergence Across Methods')
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='gray',
              fancybox=True, shadow=False)
    ax.set_xlim(0, 5000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    fig.tight_layout()
    save_fig(fig, 'fig5_reward_4way')
    plt.close(fig)


# ============================================================================
# FIGURE 6: Ablation Waterfall (contribution of each component)
# ============================================================================

def fig6_ablation_waterfall():
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Converged decode rates
    random_dec  = np.mean(ALL['random']['decode_rates'][-500:])
    ppo_dec     = np.mean(ALL['ppo_only']['decode_rates'][-500:])
    proposed_dec = np.mean(ALL['proposed']['decode_rates'][-500:])
    ideal_dec   = np.mean(ALL['ideal']['decode_rates'][-500:])

    categories = ['Random\nBaseline', '+PPO\nTrajectory', '+SAC RIS\nOptimization',
                   '+Ideal\nHardware']
    values     = [random_dec, ppo_dec, proposed_dec, ideal_dec]
    increments = [random_dec,
                  ppo_dec - random_dec,
                  proposed_dec - ppo_dec,
                  ideal_dec - proposed_dec]

    bar_colors = [COLORS['random'], COLORS['ppo_only'], COLORS['proposed'], COLORS['ideal']]

    # Stacked waterfall
    bottoms = [0, random_dec, ppo_dec, proposed_dec]
    bars = ax.bar(categories, increments, bottom=bottoms, width=0.55,
                  color=bar_colors, edgecolor='white', linewidth=1.5, alpha=0.88)

    # Connecting lines
    for i in range(len(categories) - 1):
        ax.plot([i + 0.275, i + 0.725], [values[i], values[i]],
                color='gray', linestyle=':', linewidth=1.0, alpha=0.6)

    # Value labels
    for i, (bar, val, inc) in enumerate(zip(bars, values, increments)):
        # Total value at top
        ax.text(i, val + 0.3, f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
        # Increment inside bar
        if inc > 0.5:
            ax.text(i, bottoms[i] + inc / 2, f'+{inc:.1f}pp',
                    ha='center', va='center', fontsize=8.5,
                    color='white', fontweight='bold')

    ax.set_ylabel('Decode Rate (%)')
    ax.set_title('Ablation Study: Contribution of Each Component')
    ax.set_ylim(0, 24)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', visible=False)

    fig.tight_layout()
    save_fig(fig, 'fig6_ablation_waterfall')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Generating 4-Way Publication Figures")
    print("=" * 70)

    fig1_aoi_convergence()
    fig2_decode_convergence()
    fig3_dual_axis()
    fig4_summary_bars()
    fig5_reward_convergence()
    fig6_ablation_waterfall()

    print("\n" + "=" * 70)
    print("ALL 6 FIGURES GENERATED SUCCESSFULLY!")
    print(f"Output directory: {FIG_DIR}")
    print("=" * 70)

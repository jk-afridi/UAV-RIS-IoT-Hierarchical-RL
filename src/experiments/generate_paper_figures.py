#!/usr/bin/env python3
"""
=============================================================================
PUBLICATION-QUALITY FIGURE GENERATOR
IEEE Transactions Style — UAV-RIS Hierarchical DRL Paper
=============================================================================
Generates 4 comparative figures:
  Fig 1: Average AoI Convergence (Proposed vs Random)
  Fig 2: Decode Rate Convergence (Proposed vs Random)
  Fig 3: Combined Dual-Axis Convergence
  Fig 4: Summary Bar Chart (Key Metrics)
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'lines.linewidth': 1.8,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

# ============================================================================
# COLOR PALETTE (IEEE-friendly, colorblind-safe)
# ============================================================================
COLORS = {
    'proposed':   '#1B4F72',
    'random':     '#C0392B',
    'no_ris':     '#7D3C98',
    'ideal_ris':  '#27AE60',
    'fill_prop':  '#AED6F1',
    'fill_rand':  '#F5B7B1',
}

# ============================================================================
# DATA LOADING
# ============================================================================
BASE = Path("/home/jawad/rsrch_two/UAV-mounted-RIS-IOT-RL/results")
OUTPUT = Path("/home/jawad/rsrch_two/UAV-mounted-RIS-IOT-RL/results/figures")
OUTPUT.mkdir(parents=True, exist_ok=True)

# Load Proposed Hierarchical DRL (Colab)
proposed = np.load(BASE / "hierarchical" / "beta0.2_rmin1 (2).25_seed42_colab" / "training_metrics.npz")
prop_avg_aois   = proposed["avg_aois"]
prop_decode     = proposed["decode_rates"]

# Load Random Baseline
random_data = np.load(BASE / "random" / "beta0.2_rmin1.25_seed42" / "training_metrics_random.npz")
rand_avg_aois   = random_data["avg_aois"]
rand_decode     = random_data["decode_rates"]

# ============================================================================
# UTILITY: Exponential Moving Average
# ============================================================================
def ema(data, alpha=0.02):
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

# Smooth the data
prop_aoi_smooth   = ema(prop_avg_aois, 0.02)
prop_dec_smooth   = ema(prop_decode * 100, 0.02)
rand_aoi_smooth   = ema(rand_avg_aois, 0.02)
rand_dec_smooth   = ema(rand_decode * 100, 0.02)

episodes = np.arange(1, len(prop_aoi_smooth) + 1)

# ============================================================================
# FIGURE 1: Average AoI Convergence
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(7, 4.5))

ax1.plot(episodes, prop_aoi_smooth, color=COLORS['proposed'],
         linewidth=2.0, label='Proposed Hierarchical DRL', zorder=3)
ax1.axhline(y=np.mean(rand_avg_aois), color=COLORS['random'],
            linewidth=2.0, linestyle='--', label='Random Baseline', zorder=2)

# Confidence band
window_size = 200
rolling_std = np.array([np.std(prop_avg_aois[max(0,i-window_size):i+1])
                        for i in range(len(prop_avg_aois))])
ax1.fill_between(episodes,
                 prop_aoi_smooth - rolling_std,
                 prop_aoi_smooth + rolling_std,
                 color=COLORS['fill_prop'], alpha=0.4, zorder=1)

ax1.set_xlabel('Training Episodes')
ax1.set_ylabel('Average Age of Information (AoI)')
ax1.set_title('(a) Average AoI Convergence', fontweight='bold')
ax1.legend(loc='upper right', frameon=True)
ax1.set_xlim([1, len(episodes)])

fig1.savefig(OUTPUT / 'fig1_avg_aoi_convergence.png')
fig1.savefig(OUTPUT / 'fig1_avg_aoi_convergence.pdf')
print(f"[OK] Figure 1 saved")
plt.close(fig1)

# ============================================================================
# FIGURE 2: Decode Rate Convergence
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(7, 4.5))

ax2.plot(episodes, prop_dec_smooth, color=COLORS['proposed'],
         linewidth=2.0, label='Proposed Hierarchical DRL', zorder=3)
ax2.axhline(y=np.mean(rand_decode) * 100, color=COLORS['random'],
            linewidth=2.0, linestyle='--', label='Random Baseline', zorder=2)

rolling_std_dec = np.array([np.std(prop_decode[max(0,i-window_size):i+1] * 100)
                            for i in range(len(prop_decode))])
ax2.fill_between(episodes,
                 prop_dec_smooth - rolling_std_dec,
                 prop_dec_smooth + rolling_std_dec,
                 color=COLORS['fill_prop'], alpha=0.4, zorder=1)

ax2.set_xlabel('Training Episodes')
ax2.set_ylabel('Successful Decode Rate (%)')
ax2.set_title('(b) Decode Rate Convergence', fontweight='bold')
ax2.legend(loc='lower right', frameon=True)
ax2.set_xlim([1, len(episodes)])

fig2.savefig(OUTPUT / 'fig2_decode_rate_convergence.png')
fig2.savefig(OUTPUT / 'fig2_decode_rate_convergence.pdf')
print(f"[OK] Figure 2 saved")
plt.close(fig2)

# ============================================================================
# FIGURE 3: Combined Dual-Axis
# ============================================================================
fig3, ax3a = plt.subplots(figsize=(7, 4.5))

ln1 = ax3a.plot(episodes, prop_aoi_smooth, color=COLORS['proposed'],
                linewidth=2.0, label='Avg AoI (Proposed)', zorder=3)
ln2 = ax3a.axhline(y=np.mean(rand_avg_aois), color=COLORS['random'],
                    linewidth=1.8, linestyle='--', label='Avg AoI (Random)')
ax3a.set_xlabel('Training Episodes')
ax3a.set_ylabel('Average Age of Information', color=COLORS['proposed'])
ax3a.tick_params(axis='y', labelcolor=COLORS['proposed'])
ax3a.set_xlim([1, len(episodes)])

ax3b = ax3a.twinx()
ln3 = ax3b.plot(episodes, prop_dec_smooth, color=COLORS['ideal_ris'],
                linewidth=2.0, label='Decode% (Proposed)', zorder=3)
ln4 = ax3b.axhline(y=np.mean(rand_decode) * 100, color='#E67E22',
                    linewidth=1.8, linestyle=':', label='Decode% (Random)')
ax3b.set_ylabel('Decode Rate (%)', color=COLORS['ideal_ris'])
ax3b.tick_params(axis='y', labelcolor=COLORS['ideal_ris'])

lines = ln1 + ln3 + [ln2, ln4]
labels = ['Avg AoI (Proposed)', 'Decode% (Proposed)',
          'Avg AoI (Random)', 'Decode% (Random)']
ax3a.legend(lines, labels, loc='center right', frameon=True)
ax3a.set_title('(c) Hierarchical DRL Training Dynamics', fontweight='bold')

fig3.savefig(OUTPUT / 'fig3_combined_convergence.png')
fig3.savefig(OUTPUT / 'fig3_combined_convergence.pdf')
print(f"[OK] Figure 3 saved")
plt.close(fig3)

# ============================================================================
# FIGURE 4: Summary Bar Chart
# ============================================================================
fig4, axes = plt.subplots(1, 3, figsize=(10, 4))

methods = ['Proposed\nHierarchical DRL', 'Random\nBaseline']
bar_colors = [COLORS['proposed'], COLORS['random']]

prop_final_aoi = np.mean(prop_avg_aois[-50:])
rand_final_aoi = np.mean(rand_avg_aois)
prop_final_dec = np.mean(prop_decode[-50:]) * 100
rand_final_dec = np.mean(rand_decode) * 100
prop_best_dec = np.max(prop_decode) * 100
rand_best_dec = np.max(rand_decode) * 100

# Bar 1: Average AoI
bars1 = axes[0].bar(methods, [prop_final_aoi, rand_final_aoi],
                     color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel('Average AoI')
axes[0].set_title('Avg AoI (lower is better)', fontweight='bold', fontsize=11)
for bar, val in zip(bars1, [prop_final_aoi, rand_final_aoi]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, max(prop_final_aoi, rand_final_aoi) * 1.15)

# Bar 2: Average Decode Rate
bars2 = axes[1].bar(methods, [prop_final_dec, rand_final_dec],
                     color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Decode Rate (%)')
axes[1].set_title('Avg Decode (higher is better)', fontweight='bold', fontsize=11)
for bar, val in zip(bars2, [prop_final_dec, rand_final_dec]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylim(0, max(prop_final_dec, rand_final_dec) * 1.25)

# Bar 3: Peak Decode Rate
bars3 = axes[2].bar(methods, [prop_best_dec, rand_best_dec],
                     color=bar_colors, width=0.5, edgecolor='black', linewidth=0.5)
axes[2].set_ylabel('Peak Decode Rate (%)')
axes[2].set_title('Peak Decode (higher is better)', fontweight='bold', fontsize=11)
for bar, val in zip(bars3, [prop_best_dec, rand_best_dec]):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[2].set_ylim(0, max(prop_best_dec, rand_best_dec) * 1.25)

fig4.suptitle('(d) Performance Summary: Proposed vs Random Baseline',
              fontweight='bold', fontsize=13, y=1.02)
fig4.tight_layout()

fig4.savefig(OUTPUT / 'fig4_summary_bars.png')
fig4.savefig(OUTPUT / 'fig4_summary_bars.pdf')
print(f"[OK] Figure 4 saved")
plt.close(fig4)

print("\n" + "=" * 60)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print(f"Output directory: {OUTPUT}")
print("=" * 60)

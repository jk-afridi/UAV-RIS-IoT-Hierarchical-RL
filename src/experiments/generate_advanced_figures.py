#!/usr/bin/env python3
"""
=============================================================================
ADVANCED RIGOROUS EVALUATION SCRIPT
=============================================================================
This script performs a deep-dive evaluation of the trained Beta-Space HDRL model.
It loads the exact trained weights and runs ONE deterministic evaluation episode.
During this episode, it extracts rigorous physical telemetry to plot 4 new
high-quality figures for the paper:
  Fig 7: 2D UAV Flight Trajectory over IoT clusters
  Fig 8: Instantaneous AoI Sawtooth waveform (Cluster tracking)
  Fig 9: UAV Energy Depletion curve
  Fig 10: Beta-Space RIS Amplitude Distribution Heatmap
=============================================================================
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from core.aoi_environment import AoIUAVRISEnvironment
from core import utils
from train.train_hierarchical import PPOMacroAgent, LowLevelSACWrapper

# ============================================================================
# IEEE STYLE CONFIGURATION
# ============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
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
    'axes.linewidth': 0.8,
})

def main():
    print("="*80)
    print("STARTING DEEP DIVE RIGOROUS EVALUATION")
    print("="*80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize exactly identical environment
    env = AoIUAVRISEnvironment(
        num_antennas=4,
        num_RIS_elements=64,
        num_users=4,
        beta_min=0.2,
        AWGN_var=1e-14,         # Realistic noise floor (~ -110 dBm)
        channel_noise_var=1e-14,
        cascaded_channels=True,
        channel_est_error=True,
        seed=42,
        v_max=10.0,
        h_min=10.0,
        h_max=50.0,
        energy_budget=300000.0,
        episode_length=200,
        delta_t=1.0,
        r_min=1.25,
        k_rice=10.0,
        iot_tx_power_dbm=10.0,
    )
    env_info = env.get_env_info()

    # 2. Initialize Models
    class AoIGreedyMacroAgent:
        def __init__(self, env):
            self.env = env
            self.current_target = None

        def select_action(self):
            # Find cluster with highest average AoI
            cluster_aois = []
            for c_idx in range(self.env.num_ris_groups):
                num_users_per_cluster = len(self.env.aoi) // self.env.num_ris_groups
                start_u = c_idx * num_users_per_cluster
                end_u = start_u + num_users_per_cluster
                cluster_aois.append(np.mean(self.env.aoi[start_u:end_u]))
            
            target_cluster = np.argmax(cluster_aois)
            target_pos = self.env.ris_positions[target_cluster]
            
            # Compute velocity vector towards target
            direction = target_pos[:2] - self.env.uav_pos[:2]
            distance = np.linalg.norm(direction)
            
            if distance < 5.0:
                # Hover if close enough
                velocity = np.array([0.0, 0.0])
            else:
                velocity = (direction / distance) * 10.0 # Max speed
                
            return velocity, target_cluster

    macro_agent = AoIGreedyMacroAgent(env)
    
    micro_agent = LowLevelSACWrapper(
        env=env, device=device, beta_min=0.2, inner_steps=5, lr=1e-4
    )

    # 3. Load Trained Weights (Only Micro SAC is needed, Macro is deterministic heuristic)
    weight_dir = Path(__file__).resolve().parents[2] / 'results' / 'hierarchical' / 'beta0.2_rmin1.25_seed42'
    print(f"Loading Micro-SAC weights from: {weight_dir.name}")
    
    micro_agent.agent.actor.load_state_dict(torch.load(weight_dir / 'sac_final_actor', map_location=device, weights_only=True))
    micro_agent.agent.explorer.load_state_dict(torch.load(weight_dir / 'sac_final_explorer', map_location=device, weights_only=True))

    # 4. Rigorous Tracking Arrays
    uav_x, uav_y = [], []
    energy_levels = []
    
    # Track AoI for ALL clusters
    aoi_all_clusters = {i: [] for i in range(env.num_ris_groups)}
    served_clusters_log = []
    all_betas = []
    
    # 5. Run Exact Evaluation Episode
    high_state = env.high_level_reset()
    cluster_centers = env.ris_positions # Save for plotting
    
    micro_agent.agent.actor.eval()
    micro_agent.agent.explorer.eval()

    print("Running 200 time steps...")
    for t in range(env.episode_length):
        # Tracking
        uav_x.append(env.uav_pos[0].copy())
        uav_y.append(env.uav_pos[1].copy())
        energy_levels.append(env.energy_remaining)
        
        for c_idx in range(env.num_ris_groups):
            num_users_per_cluster = env_info['K_total'] // env.num_ris_groups
            start_u = c_idx * num_users_per_cluster
            end_u = start_u + num_users_per_cluster
            aoi_all_clusters[c_idx].append(np.mean(env.aoi[start_u:end_u]))

        # Macro step (AoI Greedy Scheduler)
        velocity, cluster_id = macro_agent.select_action()
        served_clusters_log.append(cluster_id)
        
        next_high_state, done, info = env.high_level_step(velocity, cluster_id)
        low_state = info['low_state']
        
        # Micro step (Deterministic)
        with torch.no_grad():
            action_low_batch, _ = micro_agent.agent.select_action(utils.whiten(low_state), exp_regularization=0.0, evaluate=True)
            action_low = action_low_batch[0] if len(action_low_batch.shape) > 1 else action_low_batch
            
            # Compute ACTUAL synthesized Beta by parsing the hardware-constrained phase
            Phi_real = action_low[-2 * env.L:-env.L]
            Phi_imag = action_low[-env.L:]
            angles = np.arctan2(Phi_real, Phi_imag)  # same argument order as core/environment.py (was swapped -> wrong beta in Fig.10)
            actual_betas = env._compute_PDA(angles)
            all_betas.append(actual_betas) # Save the 64 mapped amplitude values
        
        # Environment step
        next_state, _, done, info = env.step(action_low_batch)
        rates = env.compute_rates_from_action(action_low)
        reward_high = env.finalize_step(cluster_id, rates)
        high_state = env.get_high_level_state()
        
        if done and env.energy_remaining <= 0:
            print(f"Energy depleted early at step {t}")
            break

    print("Evaluation Complete. Generating Figures...")
    
    fig_dir = Path(__file__).resolve().parents[2] / 'results' / 'figures_advanced'
    fig_dir.mkdir(parents=True, exist_ok=True)

    time_steps = np.arange(len(uav_x))
    
    # Identify the most frequently served cluster
    from collections import Counter
    cluster_counts = Counter(served_clusters_log)
    most_visited = cluster_counts.most_common(1)[0][0]
    least_visited = cluster_counts.most_common()[-1][0] if len(cluster_counts) > 1 else (most_visited - 1) % env.num_ris_groups

    # ---------------------------------------------------------
    # FIGURE 7: 2D UAV TRAJECTORY
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(uav_x, uav_y, '-', color='#1B4F72', lw=2, label='UAV Flight Path')
    ax.scatter(uav_x[0], uav_y[0], marker='s', s=100, color='green', label='Start', zorder=5)
    ax.scatter(uav_x[-1], uav_y[-1], marker='X', s=100, color='red', label='End', zorder=5)
    
    for i, c in enumerate(cluster_centers):
        ax.scatter(c[0], c[1], marker='^', s=120, color='#D35400', zorder=4, 
                   label='IoT Cluster' if i==0 else "")
        circle = Circle((c[0], c[1]), 10, color='#D35400', alpha=0.1)
        ax.add_patch(circle)

    ax.set_xlim(0, env.city_size)
    ax.set_ylim(0, env.city_size)
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_title('UAV 2D Trajectory (AoI-greedy macro-controller, NOT the learned PPO policy)', fontsize=10)
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig7_trajectory.pdf')
    fig.savefig(fig_dir / 'fig7_trajectory.png', dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # FIGURE 8: INSTANTANEOUS AOI SAWTOOTH
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_steps, aoi_all_clusters[most_visited], '-', color='#E74C3C', lw=1.5, label=f'Served Cluster {most_visited+1}')
    ax.plot(time_steps, aoi_all_clusters[least_visited], '-', color='#2980B9', lw=1.5, label=f'Unserved Cluster {least_visited+1}')
    
    ax.set_xlabel('Time Step ($t$)')
    ax.set_ylabel('Instantaneous AoI')
    ax.set_title('AoI evolution under the AoI-greedy macro-controller', fontsize=11)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig8_aoi_sawtooth.pdf')
    fig.savefig(fig_dir / 'fig8_aoi_sawtooth.png', dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # FIGURE 9: ENERGY DEPLETION
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_steps, np.array(energy_levels)/1000, '-', color='#27AE60', lw=2.5)
    ax.fill_between(time_steps, np.array(energy_levels)/1000, 0, color='#27AE60', alpha=0.2)
    
    ax.set_xlabel('Time Step ($t$)')
    ax.set_ylabel('Remaining Energy (kJ)')
    ax.set_title('UAV Propulsion Energy Depletion')
    ax.set_ylim(0, env.energy_budget / 1000 + 10)
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig9_energy.pdf')
    fig.savefig(fig_dir / 'fig9_energy.png', dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------
    # FIGURE 10: BETA (AMPLITUDE) HEATMAP / DISTRIBUTION
    # ---------------------------------------------------------
    beta_matrix = np.array(all_betas)
    print(f"beta_matrix shape: {beta_matrix.shape}")
    print(f"beta_matrix.T shape: {beta_matrix.T.shape}")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    cax = ax.imshow(beta_matrix.T, aspect='auto', cmap='viridis', 
                    vmin=0.2, vmax=1.0, interpolation='nearest')
    
    ax.set_xlabel('Time Step ($t$)')
    ax.set_ylabel('RIS Element Index ($l$)')
    ax.set_title(r'Heatmap of Explored Phase-Dependent Amplitudes ($\beta$)')
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(r'Amplitude $\beta^{(l)}$')
    
    fig.tight_layout()
    fig.savefig(fig_dir / 'fig10_beta_heatmap.pdf')
    fig.savefig(fig_dir / 'fig10_beta_heatmap.png', dpi=300)
    plt.close(fig)

    print("="*80)
    print("SUCCESS: 4 rigorous figures generated in results/figures_advanced/")
    print("="*80)

if __name__ == "__main__":
    main()

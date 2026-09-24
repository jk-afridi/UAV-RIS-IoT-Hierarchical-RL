import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add src/ to path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.aoi_environment import AoIUAVRISEnvironment

def train_random(args):
    # Initialize Environment
    env = AoIUAVRISEnvironment(
        num_antennas=4,
        num_RIS_elements=64,
        num_users=4,
        beta_min=args.beta_min,
        AWGN_var=1e-14,
        channel_noise_var=1e-14,
        cascaded_channels=True,
        channel_est_error=False,
        seed=args.seed,
        v_max=10.0,
        h_min=10.0,
        h_max=50.0,
        energy_budget=300000.0,
        episode_length=args.episode_length,
        delta_t=1.0,
        r_min=args.r_min,
        k_rice=10.0,
        iot_tx_power_dbm=10.0
    )

    env_info = env.get_env_info()

    # Create results directory
    results_dir = os.path.join(
        SRC_DIR.parent, 'results', 'random',
        f'beta{args.beta_min}_rmin{args.r_min}_seed{args.seed}'
    )
    os.makedirs(results_dir, exist_ok=True)

    # Training metrics
    episode_rewards = []
    episode_avg_aois = []
    episode_peak_aois = []
    episode_decode_rates = []

    print(f"\nRunning RANDOM BASELINE for {args.episodes} episodes...")
    print(f"Results will be saved to: {results_dir}")
    print("-" * 70)

    start_time = time.time()

    for episode in range(args.episodes):
        # Reset environment
        high_state = env.high_level_reset()
        episode_reward = 0
        step_aois = []
        step_decode_counts = []

        for t in range(args.episode_length):
            # STEP 1: Random High-Level Action (Velocity & Cluster)
            velocity = np.random.uniform(-1.0, 1.0, size=2)
            cluster_id = np.random.randint(0, env.num_ris_groups)

            # STEP 2: Environment moves the drone
            next_high_state, done, info = env.high_level_step(velocity, cluster_id)

            # STEP 3: Random Low-Level Action (Beamforming & RIS Phase)
            # action_dim = 2 * M^2 + 2 * L = 2*16 + 2*64 = 160
            action_dim = 2 * (env.M ** 2) + 2 * env.L
            random_action = np.random.uniform(-1.0, 1.0, size=action_dim)
            
            # Compute rates directly
            achieved_rates = env.compute_rates_from_action(random_action)

            # STEP 4: Environment updates AoI based on achieved rates
            reward = env.finalize_step(cluster_id, achieved_rates)

            episode_reward += reward
            step_aois.append(env.get_average_aoi())
            step_decode_counts.append(np.sum(achieved_rates >= args.r_min))

            if done:
                break

        # Calculate metrics
        avg_aoi = np.mean(step_aois)
        peak_aoi = env.get_peak_aoi()
        decode_pct = np.sum(step_decode_counts) / (args.episode_length * env.K)

        episode_rewards.append(episode_reward)
        episode_avg_aois.append(avg_aoi)
        episode_peak_aois.append(peak_aoi)
        episode_decode_rates.append(decode_pct)

        # Print progress
        if (episode + 1) % 1 == 0:
            print(f"  Ep {episode+1:5d}/{args.episodes} | Reward: {episode_reward:8.2f} | "
                  f"Avg AoI: {avg_aoi:6.1f} | Peak AoI: {peak_aoi:6.1f} | "
                  f"Decode%: {100*decode_pct:5.1f}% | Time: {int(time.time() - start_time)}s")

    # Save metrics
    metrics_file = os.path.join(results_dir, "training_metrics_random.npz")
    np.savez(metrics_file,
             rewards=np.array(episode_rewards),
             avg_aois=np.array(episode_avg_aois),
             peak_aois=np.array(episode_peak_aois),
             decode_rates=np.array(episode_decode_rates))
    
    print("=" * 70)
    print("RANDOM BASELINE COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {int(time.time() - start_time)}s")
    print(f"  Final Avg AoI: {np.mean(episode_avg_aois[-50:]):.2f}")
    print(f"  Final Peak AoI: {np.mean(episode_peak_aois[-50:]):.2f}")
    print(f"  Final Decode%: {np.mean(episode_decode_rates[-50:])*100:.1f}%")
    print(f"  Metrics saved to {metrics_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Baseline for AoI-Aware UAV-RIS')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes (default: 1000)')
    parser.add_argument('--episode_length', type=int, default=50,
                        help='Timesteps per episode (default: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--beta_min', type=float, default=0.2,
                        help='PDA severity (default: 0.2, cheap hardware)')
    parser.add_argument('--r_min', type=float, default=1.25,
                        help='Decode threshold in bps/Hz (default: 1.25)')
    args = parser.parse_args()
    train_random(args)

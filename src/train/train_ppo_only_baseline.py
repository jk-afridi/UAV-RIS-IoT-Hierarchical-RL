#!/usr/bin/env python3
"""
=============================================================================
BASELINE: PPO-Only (No RIS Optimization)
=============================================================================
The UAV trajectory is learned by PPO, but the RIS phase shifts and
beamforming vectors are RANDOM (no SAC micro-optimizer).

This proves that the Low-Level SAC agent is essential for achieving
high decode rates — trajectory optimization alone is insufficient.

USAGE:
    cd /home/jawad/rsrch_two/UAV-mounted-RIS-IOT-RL/src
    python -m train.train_ppo_only_baseline --episodes 5000 --seed 42 --gpu 0
=============================================================================
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

# Add src/ to path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.aoi_environment import AoIUAVRISEnvironment


# ============================================================================
# PPO AGENT (copied from train_hierarchical.py)
# ============================================================================

class PPOMacroActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = torch.tanh(self.mean_head(x))
        std = torch.clamp(self.log_std.exp(), 0.01, 1.0)
        return mean, std

    def get_action(self, state):
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1, 1), log_prob

    def evaluate(self, states, actions):
        mean, std = self.forward(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy


class PPOMacroCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class PPOMacroAgent:
    def __init__(self, state_dim, vel_dim, num_clusters,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64,
                 device='cpu'):
        self.vel_dim = vel_dim
        self.num_clusters = num_clusters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.device = device
        self.action_dim = vel_dim + 1

        self.actor = PPOMacroActor(state_dim, self.action_dim).to(device)
        self.critic = PPOMacroCritic(state_dim).to(device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []

    def select_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_t).item()
            action, log_prob = self.actor.get_action(state_t)
        action = action.squeeze(0).cpu().numpy()
        log_prob = log_prob.item()
        velocity = action[:self.vel_dim] * 10.0
        cluster_raw = action[self.vel_dim]
        cluster_id = int((cluster_raw + 1) / 2 * self.num_clusters)
        cluster_id = np.clip(cluster_id, 0, self.num_clusters - 1)
        return velocity, cluster_id, action, log_prob, value

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value):
        advantages, gae = [], 0
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = (self.rewards[t] +
                     self.gamma * values[t + 1] * (1 - self.dones[t]) -
                     values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def update(self, next_value):
        if len(self.states) == 0:
            return 0.0
        advantages, returns = self.compute_gae(next_value)
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss, n_updates = 0.0, 0
        for _ in range(self.ppo_epochs):
            indices = np.arange(len(self.states))
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages_t[batch_idx]
                batch_returns = returns_t[batch_idx]

                new_log_probs, entropy = self.actor.evaluate(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (torch.clamp(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * batch_advantages)
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -0.01 * entropy.mean()

                values_pred = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values_pred, batch_returns)

                self.actor_optimizer.zero_grad()
                (actor_loss + entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

                total_loss += (actor_loss.item() + critic_loss.item())
                n_updates += 1

        self.states.clear(); self.actions.clear(); self.log_probs.clear()
        self.rewards.clear(); self.dones.clear(); self.values.clear()
        return total_loss / max(n_updates, 1)


# ============================================================================
# MAIN TRAINING LOOP (PPO trajectory + RANDOM RIS)
# ============================================================================

def train_ppo_only(args):
    print("=" * 70)
    print("BASELINE: PPO-Only (No RIS Optimization)")
    print("UAV trajectory learned, RIS phases RANDOM")
    print("=" * 70)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = AoIUAVRISEnvironment(
        num_antennas=4, num_RIS_elements=64, num_users=4,
        beta_min=args.beta_min, AWGN_var=1e-14, channel_noise_var=1e-14,
        cascaded_channels=True, channel_est_error=True, seed=args.seed, layout_seed=args.seed, k_rice_bs=1.0,
        v_max=10.0, h_min=10.0, h_max=50.0, energy_budget=300000.0,
        episode_length=args.episode_length, delta_t=1.0, r_min=args.r_min,
        k_rice=10.0, iot_tx_power_dbm=10.0
    )

    env_info = env.get_env_info()
    action_dim = 2 * (env.M ** 2) + 2 * env.L  # low-level action size

    print(f"\nEnvironment: {env_info['K_total']} devices, "
          f"{env_info['num_clusters']} clusters, "
          f"{env_info['L']} RIS elements")

    # Create PPO agent (trajectory only)
    ppo_agent = PPOMacroAgent(
        state_dim=env_info['high_state_dim'],
        vel_dim=2, num_clusters=env_info['num_clusters'],
        lr=args.ppo_lr, gamma=0.99, gae_lambda=0.95,
        clip_epsilon=0.2, ppo_epochs=10,
        mini_batch_size=min(64, args.episode_length),
        device=device,
    )

    results_dir = os.path.join(
        SRC_DIR.parent, 'results', 'ppo_only',
        f'beta{args.beta_min}_rmin{args.r_min}_seed{args.seed}'
    )
    os.makedirs(results_dir, exist_ok=True)

    episode_rewards, episode_avg_aois = [], []
    episode_peak_aois, episode_decode_rates = [], []

    print(f"\nTraining for {args.episodes} episodes...")
    print(f"Results will be saved to: {results_dir}")
    print("-" * 70)

    start_time = time.time()

    for episode in range(args.episodes):
        high_state = env.high_level_reset()
        episode_reward = 0
        step_aois, step_decode_counts = [], []

        for t in range(args.episode_length):
            # PPO picks trajectory and cluster (LEARNED)
            velocity, cluster_id, raw_action, log_prob, value = \
                ppo_agent.select_action(high_state)

            # Environment moves the drone
            next_high_state, done, info = env.high_level_step(velocity, cluster_id)

            # RIS phases are RANDOM (no SAC optimization)
            random_action = np.random.uniform(-1.0, 1.0, size=action_dim)
            achieved_rates = env.compute_rates_from_action(random_action)

            # Update AoI
            reward = env.finalize_step(cluster_id, achieved_rates)

            ppo_agent.store_transition(
                high_state, raw_action, log_prob, reward, done, value
            )

            episode_reward += reward
            step_aois.append(env.get_average_aoi())
            step_decode_counts.append(np.sum(achieved_rates >= env.r_min))

            high_state = next_high_state
            if done:
                break

        # PPO update
        next_value_t = torch.FloatTensor(
            [0.0] if done else
            [ppo_agent.critic(
                torch.FloatTensor(high_state).unsqueeze(0).to(device)
            ).item()]
        ).item()
        ppo_loss = ppo_agent.update(next_value_t)

        # Record metrics
        avg_aoi = np.mean(step_aois)
        peak_aoi = env.get_peak_aoi()
        avg_decode = np.mean(step_decode_counts) / env.K * 100

        episode_rewards.append(episode_reward)
        episode_avg_aois.append(avg_aoi)
        episode_peak_aois.append(peak_aoi)
        episode_decode_rates.append(avg_decode)

        elapsed = time.time() - start_time
        print(f"  Ep {episode+1:>5}/{args.episodes} | "
              f"Reward: {episode_reward:>8.2f} | "
              f"Avg AoI: {avg_aoi:>6.1f} | "
              f"Peak AoI: {peak_aoi:>6.1f} | "
              f"Decode%: {avg_decode:>5.1f}% | "
              f"PPO Loss: {ppo_loss:>7.4f} | "
              f"Time: {elapsed:.0f}s")

        # Save checkpoints
        if (episode + 1) % 100 == 0:
            np.savez(
                os.path.join(results_dir, 'training_metrics.npz'),
                rewards=episode_rewards, avg_aois=episode_avg_aois,
                peak_aois=episode_peak_aois, decode_rates=episode_decode_rates,
            )

    # Final save
    total_time = time.time() - start_time
    np.savez(
        os.path.join(results_dir, 'training_metrics.npz'),
        rewards=episode_rewards, avg_aois=episode_avg_aois,
        peak_aois=episode_peak_aois, decode_rates=episode_decode_rates,
    )

    print("=" * 70)
    print("PPO-ONLY BASELINE COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final Avg AoI: {np.mean(episode_avg_aois[-100:]):.2f}")
    print(f"  Final Decode%: {np.mean(episode_decode_rates[-100:]):.1f}%")
    print(f"  Results: {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PPO-Only Baseline (No RIS Optimization)')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--episode_length', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--beta_min', type=float, default=0.2)
    parser.add_argument('--r_min', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=3e-4)
    args = parser.parse_args()
    train_ppo_only(args)

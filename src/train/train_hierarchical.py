#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
HIERARCHICAL DRL TRAINING: AoI-Aware UAV-RIS IoT Data Collection
=============================================================================

This script implements the two-tier Hierarchical DRL training loop:

  HIGH-LEVEL (PPO Macro-Planner):
    - Observes: UAV position, all AoI values, remaining energy
    - Decides: velocity vector (where to fly) + cluster selection (who to serve)
    - Reward: negative average AoI (wants data to stay fresh)

  LOW-LEVEL (Beta-Space SAC Micro-Optimizer):
    - Observes: channel matrices for the selected cluster
    - Decides: RIS phase shifts + beamforming vectors
    - Reward: uplink sum-rate (wants maximum data throughput)

The two agents communicate:
  1. PPO picks velocity + cluster → environment moves drone
  2. SAC sees the new channel state → optimizes mirror angles
  3. Environment computes uplink rates → updates AoI → returns reward to PPO

USAGE:
    cd /home/jawad/rsrch_two/UAV-mounted-RIS-IOT-RL/src
    python -m train.train_hierarchical --episodes 5000 --seed 42

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
from core import utils


# ============================================================================
# HIGH-LEVEL PPO AGENT (Macro-Planner for Trajectory + Cluster Selection)
# ============================================================================

class PPOMacroActor(nn.Module):
    """PPO Actor for the high-level macro-planner."""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
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
    """PPO Critic (value function) for the high-level agent."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state)


class PPOMacroAgent:
    """
    The High-Level PPO Agent (Macro-Planner).

    Responsible for:
      - Choosing UAV velocity (continuous, 2D)
      - Choosing which IoT cluster to serve (discrete, mapped from continuous)
    """

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

        # Action dim = velocity (2D) + cluster selection (1D, continuous → discrete)
        self.action_dim = vel_dim + 1

        self.actor = PPOMacroActor(state_dim, self.action_dim).to(device)
        self.critic = PPOMacroCritic(state_dim).to(device)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        # Trajectory buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state):
        """Select action given state. Returns velocity and cluster_id."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            value = self.critic(state_t).item()
            action, log_prob = self.actor.get_action(state_t)

        action = action.squeeze(0).cpu().numpy()
        log_prob = log_prob.item()

        # Split action into velocity and cluster selection
        velocity = action[:self.vel_dim] * 10.0  # Scale to v_max = 10 m/s
        cluster_raw = action[self.vel_dim]         # In [-1, 1]
        cluster_id = int((cluster_raw + 1) / 2 * self.num_clusters)
        cluster_id = np.clip(cluster_id, 0, self.num_clusters - 1)

        return velocity, cluster_id, action, log_prob, value

    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store a transition for PPO update."""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
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
        """Run PPO update on collected trajectory."""
        if len(self.states) == 0:
            return 0.0

        advantages, returns = self.compute_gae(next_value)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            # Mini-batch sampling
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

                # Actor loss (clipped surrogate)
                new_log_probs, entropy = self.actor.evaluate(
                    batch_states, batch_actions
                )
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                surr1 = ratio * batch_advantages
                surr2 = (torch.clamp(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) *
                         batch_advantages)
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -0.01 * entropy.mean()

                # Critic loss
                values_pred = self.critic(batch_states).squeeze()
                critic_loss = F.mse_loss(values_pred, batch_returns)

                # Combined update
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

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return total_loss / max(n_updates, 1)

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)
        print(f"  [PPO] Model saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        print(f"  [PPO] Model loaded from {path}")


# ============================================================================
# LOW-LEVEL SAC WRAPPER (Micro-Optimizer for RIS Phase Shifts)
# ============================================================================

class LowLevelSACWrapper:
    """
    Wrapper around the existing Beta_Space_Exp_SAC for the low-level agent.

    In each high-level timestep, this agent runs multiple inner iterations
    to optimize the RIS phase shifts for the currently selected cluster.
    """

    def __init__(self, env, device, beta_min=0.2, inner_steps=5,
                 lr=1e-4, buffer_size=20000, batch_size=16, use_beta=True):
        from algorithms import Beta_Space_Exp_SAC

        self.env = env
        self.device = device
        self.inner_steps = inner_steps
        self.batch_size = batch_size

        # Initialize the Beta-Space SAC agent
        self.agent = Beta_Space_Exp_SAC.Beta_Space_Exp_SAC(
            state_dim=env.state_dim,
            action_space=env.action_space,
            M=env.M,
            N=env.L,
            K=env.K,
            power_t=env.iot_tx_power_dbm,
            actor_lr=lr,
            critic_lr=lr,
            policy_type="Gaussian",
            alpha=0.2,
            target_update_interval=1,
            automatic_entropy_tuning=True,
            device=device,
            discount=0.99,
            tau=1e-3,
            beta_min=beta_min,
            use_beta=use_beta,
        )

        # Replay buffer
        self.replay_buffer = utils.BetaExperienceReplayBuffer(
            env.state_dim, env.action_dim, env.L, buffer_size
        )
        self.replay_buffer.device = device

        self.exp_reg = 0.3  # Exploration regularization
        self.total_steps = 0

    def optimize_for_cluster(self, low_state):
        """
        Run the low-level SAC for `inner_steps` to optimize the RIS
        phase shifts for the currently targeted cluster.

        Args:
            low_state: The channel state from aoi_environment.reset()

        Returns:
            best_rates: The achieved uplink rates for the 4 users
            best_action: The best action found
        """
        state = utils.whiten(low_state)
        best_reward = -np.inf
        best_rates = np.zeros(self.env.K)
        best_action = None

        for step in range(self.inner_steps):
            # Select action using Beta-Space exploration
            action, beta = self.agent.select_action(state, self.exp_reg)

            # Step the low-level environment to get next_state
            next_state, _, done, info = self.env.step(action)
            next_state_w = utils.whiten(next_state)

            # Use our CORRECT uplink rates as the reward
            rates = self.env.compute_rates_from_action(action)
            reward = np.sum(rates)

            # Reward shaping: Massive bonus for crossing the decode threshold
            decode_bonus = np.sum(rates >= self.env.r_min) * 50.0
            reward += decode_bonus

            # Store experience
            self.replay_buffer.add(
                state, action, beta, next_state_w,
                reward, float(done)
            )

            # Train if buffer has enough samples
            # Optimize for CPU: Run heavy neural net backprop every 50 steps (to keep it fast)
            if self.replay_buffer.size > self.batch_size and self.total_steps % 50 == 0:
                self.agent.update_parameters(
                    self.replay_buffer, self.exp_reg, self.batch_size
                )

            # Track best
            if reward > best_reward:
                best_reward = reward
                best_rates = rates
                best_action = action

            state = next_state_w
            self.total_steps += 1

        # Decay exploration over time
        self.exp_reg = max(0.01, self.exp_reg * 0.9999)

        return best_rates, best_action

    def act_deterministic(self, low_state):
        """Single deterministic policy action -- NO oracle best-of-N selection.
        Use this (not optimize_for_cluster) when reporting evaluation results,
        otherwise the learned policy is compared against baselines that do not
        get the same free true-rate feedback."""
        state = utils.whiten(low_state)
        action, _ = self.agent.select_action(state, 0.0, evaluate=True)
        return action[0] if getattr(action, 'ndim', 1) > 1 else action

    def save(self, path):
        self.agent.save(path)
        print(f"  [SAC] Model saved to {path}")

    def load(self, path):
        self.agent.load(path)
        print(f"  [SAC] Model loaded from {path}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_hierarchical(args):
    """Main hierarchical training loop."""

    print("=" * 70)
    print("HIERARCHICAL DRL TRAINING")
    print("AoI-Aware UAV-RIS IoT Data Collection")
    print("=" * 70)

    # Device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = AoIUAVRISEnvironment(
        num_antennas=4,
        num_RIS_elements=64,
        num_users=4,
        beta_min=args.beta_min,
        AWGN_var=1e-14,         # Realistic noise floor (~ -110 dBm)
        channel_noise_var=1e-14,
        cascaded_channels=True,
        channel_est_error=True,
        seed=args.seed,
        v_max=10.0,
        h_min=10.0,
        h_max=50.0,
        energy_budget=300000.0,
        episode_length=args.episode_length,
        delta_t=1.0,
        r_min=args.r_min,
        k_rice=10.0,
        k_rice_bs=args.k_rice_bs,
        iot_tx_power_dbm=args.power_dbm,
        layout_seed=args.seed,
    )

    env_info = env.get_env_info()
    print(f"\nEnvironment Configuration:")
    print(f"  Total IoT devices: {env_info['K_total']}")
    print(f"  Clusters: {env_info['num_clusters']}")
    print(f"  RIS elements: {env_info['L']}")
    print(f"  BS antennas: {env_info['M']}")
    print(f"  Episode length: {env_info['episode_length']} steps")
    print(f"  R_min (decode threshold): {env_info['r_min']} bps/Hz")
    print(f"  beta_min (PDA severity): {args.beta_min}")
    print(f"  High-level state dim: {env_info['high_state_dim']}")
    print(f"  Low-level state dim: {env_info['low_state_dim']}")

    # Create High-Level PPO Agent
    ppo_agent = PPOMacroAgent(
        state_dim=env_info['high_state_dim'],
        vel_dim=2,  # 2D velocity (x, y)
        num_clusters=env_info['num_clusters'],
        lr=args.ppo_lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=10,
        mini_batch_size=min(64, args.episode_length),
        device=device,
    )

    # Create Low-Level SAC Agent
    sac_wrapper = LowLevelSACWrapper(
        env=env,
        device=device,
        beta_min=args.beta_min,
        inner_steps=args.low_level_steps,
        lr=args.sac_lr,
        buffer_size=args.buffer_size,
        batch_size=16,
        use_beta=not args.no_beta,
    )

    # Create results directory
    results_dir = os.path.join(
        SRC_DIR.parent, 'results', 'hierarchical',
        f'beta{args.beta_min}_rmin{args.r_min}_seed{args.seed}'
    )
    os.makedirs(results_dir, exist_ok=True)
    import json, subprocess
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SRC_DIR.parent).decode().strip()
    except Exception:
        commit = 'unknown'
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(dict(args=vars(args), git_commit=commit, env_info={k: str(v) for k, v in env_info.items()}), f, indent=2)

    # Training metrics
    episode_rewards = []
    episode_avg_aois = []
    episode_peak_aois = []
    episode_decode_rates = []

    print(f"\nTraining for {args.episodes} episodes...")
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
            # ============================================================
            # STEP 1: High-Level PPO decides WHERE to fly and WHO to serve
            # ============================================================
            velocity, cluster_id, raw_action, log_prob, value = \
                ppo_agent.select_action(high_state)

            # ============================================================
            # STEP 2: Environment moves the drone and updates channels
            # ============================================================
            next_high_state, done, info = env.high_level_step(
                velocity, cluster_id
            )

            # ============================================================
            # STEP 3: Low-Level SAC optimizes the RIS for this cluster
            # ============================================================
            low_state = info['low_state']
            achieved_rates, best_action = sac_wrapper.optimize_for_cluster(
                low_state
            )

            # ============================================================
            # STEP 4: Environment updates AoI based on achieved rates
            # ============================================================
            reward = env.finalize_step(cluster_id, achieved_rates)

            # ============================================================
            # STEP 5: Store transition for PPO
            # ============================================================
            ppo_agent.store_transition(
                high_state, raw_action, log_prob, reward, done, value
            )

            # Track metrics
            episode_reward += reward
            step_aois.append(env.get_average_aoi())
            decode_success = np.sum(achieved_rates >= env.r_min)
            step_decode_counts.append(decode_success)

            high_state = next_high_state

            if done:
                break

        # ================================================================
        # END OF EPISODE: Update PPO
        # ================================================================
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

        # Print progress
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
                rewards=episode_rewards,
                avg_aois=episode_avg_aois,
                peak_aois=episode_peak_aois,
                decode_rates=episode_decode_rates,
            )
            ppo_agent.save(os.path.join(results_dir, 'ppo_checkpoint.pt'))
            sac_wrapper.save(os.path.join(results_dir, 'sac_checkpoint'))

    # ====================================================================
    # TRAINING COMPLETE
    # ====================================================================
    total_time = time.time() - start_time
    print("=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final Avg AoI: {np.mean(episode_avg_aois[-100:]):.2f}")
    print(f"  Final Peak AoI: {np.mean(episode_peak_aois[-100:]):.2f}")
    print(f"  Final Decode%: {np.mean(episode_decode_rates[-100:]):.1f}%")
    print(f"  Results: {results_dir}")

    # Final save
    np.savez(
        os.path.join(results_dir, 'training_metrics.npz'),
        rewards=episode_rewards,
        avg_aois=episode_avg_aois,
        peak_aois=episode_peak_aois,
        decode_rates=episode_decode_rates,
    )
    ppo_agent.save(os.path.join(results_dir, 'ppo_final.pt'))
    sac_wrapper.save(os.path.join(results_dir, 'sac_final'))

    return results_dir


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Hierarchical DRL Training for AoI-Aware UAV-RIS'
    )

    # Core
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of training episodes (default: 5000)')
    parser.add_argument('--episode_length', type=int, default=200,
                        help='Timesteps per episode (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')

    # Environment
    parser.add_argument('--beta_min', type=float, default=0.2,
                        help='PDA severity (default: 0.2, cheap hardware)')
    parser.add_argument('--r_min', type=float, default=0.5,
                        help='Decode threshold in bps/Hz (default: 0.5; 1.25 is infeasible in the fixed environment)')
    parser.add_argument('--k_rice_bs', type=float, default=1.0,
                        help='Rician K of the BS<->RIS link (default 1.0)')
    parser.add_argument('--power_dbm', type=float, default=10.0)
    parser.add_argument('--no_beta', action='store_true',
                        help='vanilla SAC ablation (disable Beta-space explorer scaling)')

    # PPO (High-Level)
    parser.add_argument('--ppo_lr', type=float, default=3e-4,
                        help='PPO learning rate (default: 3e-4)')

    # SAC (Low-Level)
    parser.add_argument('--sac_lr', type=float, default=1e-4,
                        help='SAC learning rate (default: 1e-4)')
    parser.add_argument('--low_level_steps', type=int, default=5,
                        help='SAC inner optimization steps per timestep (default: 5)')
    parser.add_argument('--buffer_size', type=int, default=20000,
                        help='SAC replay buffer size (default: 20000)')

    args = parser.parse_args()
    train_hierarchical(args)

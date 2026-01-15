#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from pathlib import Path
import time

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithms import SAC
from core.environment import RIS_MISO_PDA
from core.utils import whiten


def build_state(env):
    """Rebuild state from current env tensors after modifying channels."""
    init_action_g = np.hstack((np.real(env.G.reshape(1, -1)), np.imag(env.G.reshape(1, -1))))
    init_action_phi = np.hstack((
        np.real(np.diag(env.Phi)).reshape(1, -1),
        np.imag(np.diag(env.Phi)).reshape(1, -1),
    ))
    init_action = np.hstack((init_action_g, init_action_phi))

    power_t = np.real(np.diag(env.G.conjugate().T @ env.G)).reshape(1, -1) ** 2
    d_mat = env._compute_D()
    h2_tilde = env._compute_H_2_tilde(d_mat)
    power_r = np.linalg.norm(h2_tilde, axis=0).reshape(1, -1) ** 2

    if env.cascaded_channels:
        d_real = np.real(d_mat).reshape(1, -1)
        d_imag = np.imag(d_mat).reshape(1, -1)
        state = np.hstack((init_action, power_t, power_r, d_real, d_imag))
    else:
        h1_real = np.real(env.H_1).reshape(1, -1)
        h1_imag = np.imag(env.H_1).reshape(1, -1)
        h2_real = np.real(env.H_2).reshape(1, -1)
        h2_imag = np.imag(env.H_2).reshape(1, -1)
        state = np.hstack((init_action, power_t, power_r, h1_real, h1_imag, h2_real, h2_imag))

    return state


def apply_virtual_users(h2, real_indices, virtual_indices):
    """Duplicate nearest (highest-norm) real users into virtual slots."""
    norms = np.linalg.norm(h2[:, real_indices], axis=0)
    sorted_real = [idx for _, idx in sorted(zip(norms, real_indices), reverse=True)]
    new_h2 = h2.copy()

    for i, v_idx in enumerate(virtual_indices):
        source_idx = sorted_real[i % len(sorted_real)]
        new_h2[:, v_idx] = h2[:, source_idx]

    return new_h2, sorted_real


def run_experiment(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = RIS_MISO_PDA(
        num_antennas=args.num_antennas,
        num_RIS_elements=args.num_ris_elements,
        num_users=args.num_users,
        mismatch=True,
        channel_est_error=True,
        cascaded_channels=True,
        beta_min=args.beta_min,
        theta_bar=0.0,
        kappa_bar=1.5,
        AWGN_var=args.awgn_var,
        channel_noise_var=args.channel_noise_var,
        seed=args.seed,
    )

    model = SAC.SAC(
        state_dim=env.state_dim,
        action_space=env.action_space,
        M=args.num_antennas,
        N=args.num_ris_elements,
        K=args.num_users,
        power_t=args.power_t,
        actor_lr=1e-3,
        critic_lr=1e-3,
        policy_type="Gaussian",
        alpha=0.2,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        device=torch.device("cpu"),
        discount=1.0,
        tau=1e-3,
    )
    actor_path = args.model_path + "_actor"
    critic_path = args.model_path + "_critic"
    actor_opt_path = args.model_path + "_actor_optimizer"
    critic_opt_path = args.model_path + "_critic_optimizer"
    model.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
    model.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
    if os.path.exists(actor_opt_path):
        model.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location="cpu"))
    if os.path.exists(critic_opt_path):
        model.critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location="cpu"))
    model.critic_target.load_state_dict(model.critic.state_dict())

    results = []
    k_values = list(range(args.num_users, 0, -1))

    for kreal in k_values:
        rewards = []
        for _ in range(args.num_samples):
            env.reset()
            all_indices = list(range(env.K))
            real_indices = rng.choice(all_indices, size=kreal, replace=False).tolist()
            virtual_indices = [idx for idx in all_indices if idx not in real_indices]

            env.H_2, _ = apply_virtual_users(env.H_2, real_indices, virtual_indices)
            state = build_state(env)
            state = whiten(state)

            with torch.no_grad():
                action = model.select_action(state, evaluate=True)

            _, reward, _, _ = env.step(action)
            rewards.append(float(reward))

        rewards = np.array(rewards)
        results.append({
            "kreal": int(kreal),
            "ratio": float(kreal / env.K),
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std(ddof=1)),
        })

    return results


def plot_results(results, output_path):
    import matplotlib.pyplot as plt

    ratios = [r["ratio"] for r in results]
    means = [r["mean_reward"] for r in results]
    stds = [r["std_reward"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ratios, means, yerr=stds, fmt='o-', capsize=4, linewidth=2)
    ax.set_xlabel("Real User Ratio (Kreal / K)")
    ax.set_ylabel("Average Sum-Rate (bps/Hz)")
    ax.set_title("Virtual User Ratio Impact (SAC-mismatch, K=4)")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ratios)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Model path without suffix (e.g., Models/.../SAC_mismatch_0)")
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--num_antennas", type=int, default=4)
    parser.add_argument("--num_ris_elements", type=int, default=64)
    parser.add_argument("--num_users", type=int, default=4)
    parser.add_argument("--power_t", type=float, default=30.0)
    parser.add_argument("--beta_min", type=float, default=0.6)
    parser.add_argument("--awgn_var", type=float, default=1e-2)
    parser.add_argument("--channel_noise_var", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./outputs/review/virtual_user_ratio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    results = run_experiment(args)
    elapsed = time.time() - start

    json_path = os.path.join(args.output_dir, "virtual_user_ratio_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "elapsed_sec": elapsed}, f, indent=2)

    plot_path = os.path.join(args.output_dir, "virtual_user_ratio_plot.png")
    plot_results(results, plot_path)

    print(f"Saved results: {json_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()

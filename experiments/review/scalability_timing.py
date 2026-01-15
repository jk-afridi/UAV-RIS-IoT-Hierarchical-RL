#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
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


def measure_latency(model, env, iterations):
    state = env.reset()
    state = whiten(state)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model.select_action(state, evaluate=True)

    # Action latency
    action_times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            action = model.select_action(state, evaluate=True)
            action_times.append((time.perf_counter() - start) * 1000.0)

    # Step latency
    step_times = []
    action = model.select_action(state, evaluate=True)
    for _ in range(iterations):
        start = time.perf_counter()
        _ = env.step(action)
        step_times.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(action_times)), float(np.std(action_times, ddof=1)), \
        float(np.mean(step_times)), float(np.std(step_times, ddof=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root", default="./Models")
    parser.add_argument("--beta_min", type=float, default=0.6)
    parser.add_argument("--num_users", type=int, default=4)
    parser.add_argument("--num_antennas", type=int, default=4)
    parser.add_argument("--power_t", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output_dir", default="./outputs/review/scalability")
    parser.add_argument("--L_values", nargs="+", type=int, default=[16, 32, 64])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for L in args.L_values:
        model_dir = f"Beta_min. = {args.beta_min}, K = {args.num_users}, M = {args.num_antennas}, N = {L}, P_t = {float(args.power_t)}"
        model_path = os.path.join(args.model_root, model_dir, "SAC_mismatch_0")

        if not os.path.exists(model_path + "_actor"):
            print(f"Missing model: {model_path}_actor")
            continue

        env = RIS_MISO_PDA(
            num_antennas=args.num_antennas,
            num_RIS_elements=L,
            num_users=args.num_users,
            mismatch=True,
            channel_est_error=True,
            cascaded_channels=True,
            beta_min=args.beta_min,
            theta_bar=0.0,
            kappa_bar=1.5,
            AWGN_var=1e-2,
            channel_noise_var=1e-2,
            seed=0,
        )

        model = SAC.SAC(
            state_dim=env.state_dim,
            action_space=env.action_space,
            M=args.num_antennas,
            N=L,
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
        actor_path = model_path + "_actor"
        critic_path = model_path + "_critic"
        actor_opt_path = model_path + "_actor_optimizer"
        critic_opt_path = model_path + "_critic_optimizer"
        model.actor.load_state_dict(torch.load(actor_path, map_location="cpu"))
        model.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
        if os.path.exists(actor_opt_path):
            model.actor_optimizer.load_state_dict(torch.load(actor_opt_path, map_location="cpu"))
        if os.path.exists(critic_opt_path):
            model.critic_optimizer.load_state_dict(torch.load(critic_opt_path, map_location="cpu"))
        model.critic_target.load_state_dict(model.critic.state_dict())

        param_count = sum(p.numel() for p in model.actor.parameters())

        action_mean, action_std, step_mean, step_std = measure_latency(model, env, args.iterations)
        results.append({
            "L": L,
            "actor_params": param_count,
            "action_ms_mean": action_mean,
            "action_ms_std": action_std,
            "step_ms_mean": step_mean,
            "step_ms_std": step_std,
        })

        print(f"L={L} | action {action_mean:.3f}±{action_std:.3f} ms | step {step_mean:.3f}±{step_std:.3f} ms")

    csv_path = os.path.join(args.output_dir, "scalability_timing.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    plot_path = os.path.join(args.output_dir, "scalability_timing.png")
    plot_results(results, plot_path)

    print(f"Saved results: {csv_path}")
    print(f"Saved plot: {plot_path}")


def plot_results(results, output_path):
    import matplotlib.pyplot as plt

    L_values = [r["L"] for r in results]
    action_means = [r["action_ms_mean"] for r in results]
    action_stds = [r["action_ms_std"] for r in results]
    step_means = [r["step_ms_mean"] for r in results]
    step_stds = [r["step_ms_std"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(L_values, action_means, yerr=action_stds, fmt='o-', capsize=4, label="Policy Inference")
    ax.errorbar(L_values, step_means, yerr=step_stds, fmt='s--', capsize=4, label="Env Step")
    ax.set_xlabel("RIS Elements (L)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Scalability Timing (CPU)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

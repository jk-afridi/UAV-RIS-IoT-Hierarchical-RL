#!/usr/bin/env python3
"""
Train the physics-guided residual SAC (low level) on slot-level data.

    python -m src.train.train_residual --seed 0 --spread 1.0 --train_slots 30000 --device cuda \
        --out results/residual/seed0_spread1.0

One "slot" = one (cluster, UAV position, fading realisation, CSI-estimation-noise) draw. The
hardware instance (per-element PDA parameters) is FIXED for the whole run: it models one physical RIS
whose true curve differs from the nominal one by `spread` (0 = no mismatch).
The training log compares the learned policy with its own physics prior (AO under the nominal
model) on fresh slots, so "does learning add anything?" is visible directly.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.aoi_environment import AoIUAVRISEnvironment                       # noqa: E402
from baselines.ris_baselines import (cascaded_csi, ao_solution, slot_features, residual_action,  # noqa: E402
                                     feature_dim, pack_action, zf_beamformer, _amp_theta, _params, _wrap)
from algorithms.residual_sac import ResidualSAC                              # noqa: E402


def make_slot_env(seed, a):
    env = AoIUAVRISEnvironment(
        num_antennas=4, num_RIS_elements=a.L, num_users=4, beta_min=a.beta_min,
        AWGN_var=1e-14, channel_noise_var=1e-14, cascaded_channels=True, channel_est_error=True,
        seed=seed, v_max=10.0, energy_budget=3e5, episode_length=200, delta_t=1.0,
        r_min=a.r_min, k_rice=a.k_rice, k_rice_bs=a.k_rice_bs, iot_tx_power_dbm=a.power_dbm,
        layout_seed=seed)
    if a.spread > 0:
        env.set_hardware(**env.sample_hardware(np.random.default_rng(10_000 + seed), a.spread))
    return env


def sample_slot(env, rng, max_dist, fixed_dist=None):
    """Place the UAV within `max_dist` m (or exactly `fixed_dist` m) horizontally from a random
    cluster centre and redraw the channel."""
    cid = int(rng.integers(env.num_ris_groups))
    cx, cy = env.ris_positions[cid][:2]
    r = fixed_dist if fixed_dist is not None else max_dist * np.sqrt(rng.uniform())
    ang = rng.uniform(0, 2 * np.pi)
    env.set_current_ris_group(cid)
    env.uav_pos = np.array([np.clip(cx + r * np.cos(ang), 0, env.city_size),
                            np.clip(cy + r * np.sin(ang), 0, env.city_size), env.ris_height])
    env._update_channels_for_position()
    return cid


def slot_reward(rates, r_min):
    """Dense + threshold-aware: mean capped normalised rate + decoded fraction (range 0..3)."""
    return float(np.mean(np.minimum(rates / r_min, 2.0)) + np.mean(rates >= r_min))


def evaluate_vs_prior(env, agent, rng, n, max_dist, sweeps):
    """Paired evaluation on the same slots and the same CSI estimate: -> (policy, prior) mean slot reward."""
    pol, pri = [], []
    for _ in range(n):
        sample_slot(env, rng, max_dist)
        Dh = cascaded_csi(env, rng, noisy=True)
        th, G = ao_solution(env, rng, Dh=Dh, model='nominal', sweeps=sweeps)     # the prior, computed once
        pri.append(slot_reward(env.compute_rates_from_action(pack_action(env, G, th)), env.r_min))
        a, _, _ = residual_action(env, rng, agent, Dh=Dh, deterministic=True, prior_theta=th)
        pol.append(slot_reward(env.compute_rates_from_action(a), env.r_min))
    return float(np.mean(pol)), float(np.mean(pri))


def train(env, a, rng, device, log=print):
    """Trains the residual policy on PAIRED rewards (reward(prior+delta) - reward(prior), same channel and
    CSI estimate) and returns the snapshot with the best paired gain on a fixed validation stream
    (the zero-residual policy, gain 0, is the fallback), so the result can never be worse than its prior
    on validation."""
    in_dim, act_dim = feature_dim(env), env.L
    agent = ResidualSAC(in_dim, act_dim, device=device, delta_max=a.delta_max, lr=a.lr,
                        alpha_init=a.alpha, auto_alpha=a.auto_alpha, l2=a.l2, anchor_frac=a.anchor_frac)
    X = np.zeros((a.buffer, in_dim), np.float32); A = np.zeros((a.buffer, act_dim), np.float32)
    R = np.zeros(a.buffer, np.float32); n = ptr = 0
    history, t0 = [], time.time()
    slots = 0
    best_gain, best_state = 0.0, agent.snapshot()
    best_state = {k: v * 0 if k.startswith('mu.') else v for k, v in best_state.items()}   # exact zero-residual fallback
    while slots < a.train_slots:
        for _ in range(a.collect_per_iter):
            sample_slot(env, rng, a.max_dist)
            Dh = cascaded_csi(env, rng, noisy=True)
            th, G = ao_solution(env, rng, Dh=Dh, model='nominal', sweeps=a.ao_sweeps)
            r0 = slot_reward(env.compute_rates_from_action(pack_action(env, G, th)), env.r_min)
            act, x, delta = residual_action(env, rng, agent, Dh=Dh, deterministic=False, prior_theta=th)
            r = slot_reward(env.compute_rates_from_action(act), env.r_min)
            X[ptr], A[ptr], R[ptr] = x, delta / a.delta_max, r - r0     # paired reward, tanh-space action
            ptr = (ptr + 1) % a.buffer; n = min(n + 1, a.buffer); slots += 1
        if n >= a.warmup:
            for _ in range(a.updates_per_iter):
                idx = rng.integers(0, n, a.batch)
                info = agent.update(torch.as_tensor(X[idx], device=agent.device),
                                    torch.as_tensor(A[idx] * a.delta_max, device=agent.device),
                                    torch.as_tensor(R[idx], device=agent.device))
        if slots % a.eval_every < a.collect_per_iter:
            pol, pri = evaluate_vs_prior(env, agent, np.random.default_rng(777), a.eval_slots, a.max_dist, a.ao_sweeps)
            gain = pol - pri
            if gain > best_gain:
                best_gain, best_state = gain, agent.snapshot()
            history.append(dict(slots=slots, policy_reward=pol, prior_reward=pri, gain=gain, best_gain=best_gain,
                                time_s=time.time() - t0))
            log(f"slots {slots:>7d} | policy {pol:.3f} | AO-prior {pri:.3f} | gain {gain:+.3f} | best {best_gain:+.3f} "
                f"| {time.time() - t0:.0f}s")
    if a.select_best:
        agent.restore(best_state)
    return agent, history


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--spread', type=float, default=1.0, help='hardware mismatch level (0 = nominal)')
    p.add_argument('--train_slots', type=int, default=30000)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta_min', type=float, default=0.2)
    p.add_argument('--r_min', type=float, default=0.5)
    p.add_argument('--k_rice', type=float, default=10.0)
    p.add_argument('--k_rice_bs', type=float, default=1.0)
    p.add_argument('--power_dbm', type=float, default=10.0)
    p.add_argument('--max_dist', type=float, default=20.0, help='max UAV-cluster horizontal distance for slots')
    p.add_argument('--delta_max', type=float, default=1.0, help='max phase residual (rad)')
    p.add_argument('--ao_sweeps', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha', type=float, default=0.02, help='entropy weight (fixed unless --auto_alpha)')
    p.add_argument('--auto_alpha', action='store_true')
    p.add_argument('--l2', type=float, default=0.5, help='penalty on the squared normalised residual')
    p.add_argument('--anchor_frac', type=float, default=0.25, help='zero-residual anchor samples per batch')
    p.add_argument('--no_select_best', dest='select_best', action='store_false',
                   help='keep the final policy instead of the best-validation snapshot')
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--buffer', type=int, default=30000)
    p.add_argument('--warmup', type=int, default=512)
    p.add_argument('--collect_per_iter', type=int, default=8)
    p.add_argument('--updates_per_iter', type=int, default=8)
    p.add_argument('--eval_every', type=int, default=2000)
    p.add_argument('--eval_slots', type=int, default=200)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default=None)
    return p


def main():
    a = build_parser().parse_args()
    out = Path(a.out or f'results/residual/seed{a.seed}_spread{a.spread}')
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    env = make_slot_env(a.seed, a)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SRC_DIR.parent).decode().strip()
    except Exception:
        commit = 'unknown'
    (out / 'config.json').write_text(json.dumps(dict(args=vars(a), git_commit=commit), indent=2))
    agent, hist = train(env, a, rng, a.device)
    agent.save(str(out / 'residual_sac.pt'))
    (out / 'history.json').write_text(json.dumps(hist, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()

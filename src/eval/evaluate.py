#!/usr/bin/env python3
"""
Multi-seed evaluation harness (no torch needed for the non-learning methods).

Fixes the evaluation problems found in the audit of the original repo:
  * results were TRAINING curves from a single seed (42) with one fixed city layout;
  * each seed now gets its own layout (`layout_seed=seed`) and its own RNG stream;
  * every run writes a config.json (args + git commit) next to its results;
  * statistics are reported as mean +- 95% CI across seeds (t-distribution);
  * the same rate function / AoI logic scores every method;
  * episode length is 200 everywhere, so peak AoI is not saturated by a 50-step cap.

Usage (from repo root):
    python -m src.eval.evaluate --seeds 5 --episodes 2 --out results/eval_v2/baselines.json
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.aoi_environment import AoIUAVRISEnvironment          # noqa: E402
from baselines.ris_baselines import RIS_POLICIES                 # noqa: E402
from baselines.schedulers import SCHEDULERS                      # noqa: E402

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}

DEFAULT_METHODS = [
    'greedy_aoi_travel+ao_aware', 'greedy_aoi_travel+ao_unaware', 'greedy_aoi_travel+best_of_5',
    'greedy_aoi_travel+random1', 'greedy_aoi+ao_aware', 'round_robin+ao_aware', 'random+random1',
]


def make_env(a, seed):
    return AoIUAVRISEnvironment(
        num_antennas=4, num_RIS_elements=a.L, num_users=4, beta_min=a.beta_min,
        AWGN_var=1e-14, channel_noise_var=1e-14, cascaded_channels=True, channel_est_error=True,
        seed=seed, v_max=10.0, energy_budget=a.energy_budget, episode_length=a.episode_length,
        delta_t=1.0, r_min=a.r_min, k_rice=a.k_rice, k_rice_bs=a.k_rice_bs,
        iot_tx_power_dbm=a.power_dbm, layout_seed=seed)


def run_episode(env, sched, ris, rng):
    env.high_level_reset()
    avg, mx, dec, n_slots = [], 0.0, 0, 0
    for t in range(env.episode_length):
        vel, cid = sched(env, t, rng)
        _, done, info = env.high_level_step(vel, cid)
        rates = env.compute_rates_from_action(ris(env, rng))
        env.finalize_step(cid, rates)
        avg.append(env.get_average_aoi()); mx = max(mx, env.get_peak_aoi())
        dec += int(np.sum(rates >= env.r_min)); n_slots += env.K
        if done:
            break
    return dict(avg_aoi=float(np.mean(avg)), peak_aoi=float(mx), decode_pct=100.0 * dec / n_slots,
                energy_used_frac=float(1 - env.energy_remaining / env.energy_budget), steps=len(avg))


def job(args_dict, method, seed):
    a = argparse.Namespace(**args_dict)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    env = make_env(a, seed)
    sname, rname = method.split('+')
    eps = [run_episode(env, SCHEDULERS[sname], RIS_POLICIES[rname], rng) for _ in range(a.episodes)]
    return method, seed, {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), float('nan')
    df = n - 1
    t = T975.get(df) or T975[max(k for k in T975 if k <= df)] if df < 30 else 1.96
    return float(x.mean()), float(t * x.std(ddof=1) / np.sqrt(n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--episodes', type=int, default=2, help='episodes per seed')
    p.add_argument('--episode_length', type=int, default=200)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta_min', type=float, default=0.2)
    p.add_argument('--r_min', type=float, default=0.5)
    p.add_argument('--k_rice', type=float, default=10.0)
    p.add_argument('--k_rice_bs', type=float, default=1.0)
    p.add_argument('--power_dbm', type=float, default=10.0)
    p.add_argument('--energy_budget', type=float, default=300000.0)
    p.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    p.add_argument('--out', default='results/eval_v2/baselines.json')
    a = p.parse_args()

    t0 = time.time()
    jobs = [(m, s) for m in a.methods for s in range(a.seeds)]
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(job, vars(a), m, s) for m, s in jobs]
        raw = [f.result() for f in futs]

    summary = {}
    for m in a.methods:
        rows = [r for (mm, s, r) in raw if mm == m]
        summary[m] = {k: dict(zip(('mean', 'ci95'), ci([r[k] for r in rows]))) for k in rows[0]}
        summary[m]['per_seed'] = {s: r for (mm, s, r) in raw if mm == m}

    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SRC_DIR.parent).decode().strip()
    except Exception:
        commit = 'unknown'
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(a), git_commit=commit, runtime_s=time.time() - t0,
                                   results=summary), indent=2, default=str))

    print(f"{'method':32s} {'avg AoI':>16s} {'peak AoI':>16s} {'decode %':>16s}  (mean +- 95% CI over {a.seeds} seeds)")
    for m in a.methods:
        s = summary[m]
        f = lambda k: f"{s[k]['mean']:7.2f} +-{s[k]['ci95']:5.2f}"
        print(f"{m:32s} {f('avg_aoi'):>16s} {f('peak_aoi'):>16s} {f('decode_pct'):>16s}")
    print(f"saved {out}  ({time.time()-t0:.0f}s)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Episode-level study: do learned / heuristic macro policies (trajectory + scheduling) matter, and how
do they interact with the low-level RIS policy?  Schedulers x low-level policies, per-seed city layouts
(and per-seed hardware instance when --spread > 0), 200-step episodes, mean +- 95% CI over seeds.

    python -m src.eval.aoi_study --seeds 5 --spread 1.0 --lowlevels ao_nominal ao_calibrated residual \
        --residual_root results/residual --ppo_ckpt results/macro/<run>/ppo_final.pt
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.schedulers import SCHEDULERS                       # noqa: E402
from baselines.lowlevel import make_lowlevel                      # noqa: E402
from eval.evaluate import make_env, ci                            # noqa: E402


def run_episode_generic(env, sched, lowlevel, rng):
    """sched(env, t, rng) -> (velocity m/s, cluster); lowlevel(env) -> action. Same metrics as evaluate.py."""
    env.high_level_reset()
    avg, mx, dec, n = [], 0.0, 0, 0
    for t in range(env.episode_length):
        vel, cid = sched(env, t, rng)
        _, done, _ = env.high_level_step(vel, cid, build_low_state=False)
        rates = env.compute_rates_from_action(lowlevel(env))
        env.finalize_step(cid, rates)
        avg.append(env.get_average_aoi()); mx = max(mx, env.get_peak_aoi())
        dec += int(np.sum(rates >= env.r_min)); n += env.K
        if done:
            break
    return dict(avg_aoi=float(np.mean(avg)), peak_aoi=float(mx), decode_pct=100.0 * dec / n)


def ppo_scheduler(agent):
    return lambda env, t, rng: agent.env_action(env.get_high_level_state(), env.v_max, deterministic=True)


def apply_hardware(env, seed, spread):
    """Same hardware instance as train_residual.make_slot_env (so residual checkpoints match)."""
    if spread > 0:
        env.set_hardware(**env.sample_hardware(np.random.default_rng(10_000 + seed), spread))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--episode_length', type=int, default=200)
    p.add_argument('--spread', type=float, default=0.0)
    p.add_argument('--scheds', nargs='+', default=['greedy_aoi_travel', 'greedy_aoi', 'round_robin', 'random', 'ppo'])
    p.add_argument('--lowlevels', nargs='+', default=['ao_nominal', 'ao_calibrated'])
    p.add_argument('--residual_root', default=None)
    p.add_argument('--ppo_ckpt', default=None)
    p.add_argument('--ao_sweeps', type=int, default=2)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta_min', type=float, default=0.2)
    p.add_argument('--r_min', type=float, default=0.5)
    p.add_argument('--k_rice', type=float, default=10.0)
    p.add_argument('--k_rice_bs', type=float, default=1.0)
    p.add_argument('--power_dbm', type=float, default=10.0)
    p.add_argument('--energy_budget', type=float, default=300000.0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', default='results/aoi_study/aoi_study.json')
    a = p.parse_args()

    ppo = None
    if 'ppo' in a.scheds:
        from algorithms.macro_ppo import HybridPPO
        ppo = HybridPPO.load(a.ppo_ckpt, device=a.device)

    t0, per_seed = time.time(), {}
    for seed in range(a.seeds):
        for ll in a.lowlevels:
            ckpt = (Path(a.residual_root) / f'seed{seed}_spread{a.spread}' / 'residual_sac.pt') if ll == 'residual' else None
            for sname in a.scheds:
                np.random.seed(seed)
                rng = np.random.default_rng(1000 * seed + 1)
                env = make_env(a, seed); apply_hardware(env, seed, a.spread)
                sched = ppo_scheduler(ppo) if sname == 'ppo' else SCHEDULERS[sname]
                low = make_lowlevel(ll, rng, ckpt=str(ckpt) if ckpt else None, device=a.device, ao_sweeps=a.ao_sweeps)
                eps = [run_episode_generic(env, sched, low, rng) for _ in range(a.episodes)]
                per_seed[(sname, ll, seed)] = {k: float(np.mean([e[k] for e in eps])) for k in eps[0]}
        print(f'seed {seed} done ({time.time() - t0:.0f}s)', flush=True)

    summary = {}
    for ll in a.lowlevels:
        for sname in a.scheds:
            rows = [per_seed[(sname, ll, s)] for s in range(a.seeds)]
            summary[f'{sname}+{ll}'] = {k: dict(zip(('mean', 'ci95'), ci([r[k] for r in rows]))) for k in rows[0]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(config=vars(a), summary=summary,
                                   per_seed={f'{k[0]}+{k[1]}|{k[2]}': v for k, v in per_seed.items()}), indent=2, default=str))
    print(f"{'scheduler+lowlevel':36s} {'avg AoI':>16s} {'peak AoI':>16s} {'decode %':>16s}")
    for k, v in summary.items():
        f = lambda m: f"{v[m]['mean']:7.2f} +-{v[m]['ci95']:5.2f}"
        print(f"{k:36s} {f('avg_aoi'):>16s} {f('peak_aoi'):>16s} {f('decode_pct'):>16s}")
    print('saved', out)


if __name__ == '__main__':
    main()

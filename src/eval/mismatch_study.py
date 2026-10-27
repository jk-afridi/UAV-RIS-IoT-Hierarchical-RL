#!/usr/bin/env python3
"""
Hardware-mismatch study -- the experiment that decides what the paper can honestly claim.

Question: when the true RIS hardware deviates from the nominal PDA model a designer assumes,
how do methods degrade, and does a learned physics-guided residual policy recover the loss
WITHOUT per-slot probing?

Compared on identical slots / channel realisations / CSI estimates (per seed, per mismatch level):
  ao_nominal    AO assuming the nominal PDA curve        (0 probes)  <- standard model-based design
  ao_ideal      AO assuming unit-modulus reflection      (0 probes)
  ao_calibrated AO knowing the TRUE per-element curve    (0 probes)  <- ceiling; needs calibration campaign
  cem{6,15,45}  AO(nominal)-initialised CEM with B true-rate probes per slot
  best_of_45    random search, 45 probes                 <- sanity floor
  residual_sac  AO(nominal) + learned residual           (0 probes)  <- proposed (needs training/ckpt)

Metrics per slot: decode % (rate >= R_min for a device), min-user rate, mean rate.
Aggregated as mean +- 95% CI over seeds (each seed = own city layout AND own hardware instance).

    python -m src.eval.mismatch_study --seeds 5 --spreads 0 0.5 1 2 --train_slots 30000   # trains + evals
    python -m src.eval.mismatch_study --seeds 5 --skip_drl                                # baselines only
    python -m src.eval.mismatch_study --seeds 5 --ckpt_root results/residual               # reuse trained policies
"""
import copy, json, sys, time
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.ris_baselines import (cascaded_csi, ao_ris, cem_ris, best_of_n, residual_action,   # noqa: E402
                                     ao_solution, pack_action)
from eval.evaluate import ci                                                                       # noqa: E402
from train.train_residual import build_parser, make_slot_env, sample_slot                          # noqa: E402

PROBES = {'ao_nominal': 0, 'ao_ideal': 0, 'ao_calibrated': 0, 'cem6': 6, 'cem15': 15, 'cem45': 45,
          'best_of_45': 45, 'residual_sac': 0}


def methods(env, rng, Dh, agent, a):
    # ONE AO(nominal) solution per slot: it is both the 'ao_nominal' baseline and the physics prior of the
    # residual policy, so 'residual with delta=0' is exactly 'ao_nominal'.
    prior_th, prior_G = ao_solution(env, rng, Dh=Dh, model='nominal', sweeps=a.ao_sweeps)
    m = {
        'ao_nominal':    lambda: pack_action(env, prior_G, prior_th),
        'ao_ideal':      lambda: ao_ris(env, rng, Dh=Dh, model='ideal', sweeps=a.ao_sweeps),
        'ao_calibrated': lambda: ao_ris(env, rng, Dh=Dh, model='true', sweeps=a.ao_sweeps),
        'cem6':          lambda: cem_ris(env, rng, 6, Dh=Dh, model='nominal', ao_sweeps=a.ao_sweeps),
        'cem15':         lambda: cem_ris(env, rng, 15, Dh=Dh, model='nominal', ao_sweeps=a.ao_sweeps),
        'cem45':         lambda: cem_ris(env, rng, 45, Dh=Dh, model='nominal', ao_sweeps=a.ao_sweeps),
        'best_of_45':    lambda: best_of_n(env, 45, rng),
    }
    if agent is not None:
        m['residual_sac'] = lambda: residual_action(env, rng, agent, Dh=Dh, deterministic=True, prior_theta=prior_th)[0]
    return m


def run_seed_spread(a, seed, spread, log=print):
    sa = copy.copy(a); sa.spread = spread
    env = make_slot_env(seed, sa)
    agent = None
    if not a.skip_drl:
        import torch
        from algorithms.residual_sac import ResidualSAC
        ck = Path(a.ckpt_root) / f'seed{seed}_spread{spread}' / 'residual_sac.pt' if a.ckpt_root else None
        if ck is not None and ck.exists():
            agent = ResidualSAC.load(str(ck), device=a.device)
        else:
            from train.train_residual import train
            torch.manual_seed(seed)
            agent, _ = train(env, sa, np.random.default_rng(seed + 1), a.device, log=lambda *x: None)
            if a.save_root:
                d = Path(a.save_root) / f'seed{seed}_spread{spread}'; d.mkdir(parents=True, exist_ok=True)
                agent.save(str(d / 'residual_sac.pt'))
    rng = np.random.default_rng(10_000 * (seed + 1) + int(100 * spread))      # eval stream (disjoint from training)
    acc = {}
    per = max(1, a.study_slots // len(a.eval_dists))
    for d in a.eval_dists:
        for _ in range(per):
            sample_slot(env, rng, a.max_dist, fixed_dist=d)
            Dh = cascaded_csi(env, rng, noisy=True)                                # ONE CSI estimate shared by all methods
            for name, fn in methods(env, rng, Dh, agent, a).items():
                r = env.compute_rates_from_action(fn())
                acc.setdefault(name, []).append((np.mean(r >= env.r_min) * 100, r.min(), r.mean()))
    return {k: dict(decode_pct=float(np.mean([x[0] for x in v])), min_rate=float(np.mean([x[1] for x in v])),
                    mean_rate=float(np.mean([x[2] for x in v]))) for k, v in acc.items()}


def main():
    p = build_parser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--spreads', type=float, nargs='+', default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument('--study_slots', type=int, default=200, help='held-out evaluation slots per (seed, spread)')
    p.add_argument('--eval_dists', type=float, nargs='+', default=[0.0, 5.0, 10.0, 15.0])
    p.add_argument('--skip_drl', action='store_true')
    p.add_argument('--ckpt_root', default=None, help='dir with seed{S}_spread{X}/residual_sac.pt to reuse')
    p.add_argument('--save_root', default=None, help='where to save policies trained inside this script')
    p.add_argument('--study_out', default='results/mismatch_study/mismatch_study.json')
    a = p.parse_args()

    t0, raw = time.time(), {}
    for spread in a.spreads:
        for seed in range(a.seeds):
            raw[(spread, seed)] = run_seed_spread(a, seed, spread)
            print(f'  spread {spread} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)

    summary = {}
    for spread in a.spreads:
        summary[str(spread)] = {}
        names = list(raw[(spread, 0)].keys())
        for n in names:
            summary[str(spread)][n] = {'probes_per_slot': PROBES.get(n, 0)}
            for metric in ('decode_pct', 'min_rate', 'mean_rate'):
                m, c = ci([raw[(spread, s)][n][metric] for s in range(a.seeds)])
                summary[str(spread)][n][metric] = dict(mean=m, ci95=c)
    out = Path(a.study_out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(config={k: v for k, v in vars(a).items()}, summary=summary,
                                   per_seed={f'{k[0]}|{k[1]}': v for k, v in raw.items()}), indent=2, default=str))
    for spread in a.spreads:
        print(f'\n=== hardware spread {spread}  (mean +- 95% CI over {a.seeds} seeds)')
        print(f"{'method':15s} {'probes':>6s} {'decode %':>16s} {'min rate':>16s}")
        for n, v in summary[str(spread)].items():
            print(f"{n:15s} {v['probes_per_slot']:6d} {v['decode_pct']['mean']:8.1f} +-{v['decode_pct']['ci95']:5.1f} "
                  f"{v['min_rate']['mean']:8.3f} +-{v['min_rate']['ci95']:5.3f}")
    print('saved', out)


if __name__ == '__main__':
    main()

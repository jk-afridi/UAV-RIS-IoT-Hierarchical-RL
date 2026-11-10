#!/usr/bin/env python3
"""
Robustness sweeps: do the headroom findings survive other parameter settings?

One-at-a-time sweeps around the default scenario (hardware spread 1.0, R_min 0.5, K_bs 1, 10 dBm, L 64):
R_min, BS-link Rician K, IoT transmit power, RIS size L, and the decode-aware margin.
For each setting: per seed (own layout AND own hardware), same slots / CSI for all methods, paired
differences vs ao_nominal, mean +- 95% CI over seeds. Resumable: finished settings are skipped.

    python -m src.eval.sweeps --seeds 5 --study_slots 100 --out results/sweeps/sweeps.json
"""
import argparse, copy, json, sys, time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.evaluate import ci                                                  # noqa: E402
from eval.mismatch_study import run_seed_spread                               # noqa: E402
from train.train_residual import build_parser                                 # noqa: E402

METHODS = ['ao_ideal', 'ao_nominal', 'ao_calibrated', 'ao_cal0.25', 'ao_nominal_perfcsi',
           'ao_decode_nominal', 'ao_decode_cal0.25', 'ao_decode_calibrated', 'cem15', 'cem45']
# (parameter, values); the default value of each parameter is covered by the 'default' setting
GRID = [('r_min', [0.25, 1.0]), ('k_rice_bs', [0.1, 3.0]), ('power_dbm', [0.0, 20.0]), ('L', [32, 128]),
        ('margin', [0.0, 0.3, 0.5])]


def settings():
    yield 'default', {}
    for param, values in GRID:
        for v in values:
            yield f'{param}={v}', {param: v}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--spread', type=float, default=1.0)
    p.add_argument('--study_slots', type=int, default=100)
    p.add_argument('--out', default='results/sweeps/sweeps.json')
    p.add_argument('--only_settings', nargs='+', default=None, help='e.g. default L=128')
    c = p.parse_args()

    base = build_parser().parse_args([])
    base.skip_drl, base.extra, base.only = True, True, METHODS
    base.eval_dists, base.study_slots, base.ckpt_root, base.save_root, base.margin = [0.0, 5.0, 10.0, 15.0], c.study_slots, None, None, 0.15

    out = Path(c.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    t0 = time.time()
    for key, ov in settings():
        if c.only_settings and key not in c.only_settings:
            continue
        if key in res and len(res[key]['per_seed']) >= c.seeds:
            print(f'skip {key} (done)'); continue
        a = copy.copy(base)
        for k, v in ov.items():
            setattr(a, k, v)
        per_seed = [run_seed_spread(a, seed, c.spread) for seed in range(c.seeds)]
        summary = {}
        for m in per_seed[0]:
            summary[m] = {}
            for metric in ('decode_pct', 'min_rate'):
                mean, h = ci([r[m][metric] for r in per_seed]); summary[m][metric] = dict(mean=mean, ci95=h)
            if m != 'ao_nominal':
                for metric in ('decode_pct', 'min_rate'):
                    mean, h = ci([r[m][metric] - r['ao_nominal'][metric] for r in per_seed])
                    summary[m]['paired_' + metric] = dict(mean=mean, ci95=h)
        res[key] = dict(overrides=ov, spread=c.spread, per_seed=per_seed, summary=summary)
        out.write_text(json.dumps(res, indent=2))
        d = summary
        print(f"[{time.time() - t0:5.0f}s] {key:14s} ao_nominal {d['ao_nominal']['decode_pct']['mean']:5.1f} | "
              f"decode-aware {d['ao_decode_nominal']['paired_decode_pct']['mean']:+5.1f} | "
              f"cem15 {d['cem15']['paired_decode_pct']['mean']:+5.1f} | cem45 {d['cem45']['paired_decode_pct']['mean']:+5.1f}", flush=True)
    print('saved', out)


if __name__ == '__main__':
    main()

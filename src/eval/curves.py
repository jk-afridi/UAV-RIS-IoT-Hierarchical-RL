#!/usr/bin/env python3
"""
Curve experiments that turn single operating points into dose-response relations.

    calibration : decode-rate gain vs calibration error e of the assumed PDA model, for max-min AO and
                  decode-aware AO (e = 0 is the true curve; the nominal model is the e -> large limit)
    probes      : gain vs number of true-rate measurements per slot B for AO+CEM started from the nominal
                  AO solution and from the decode-aware solution, against the zero-probe decode-aware AO
    regime      : decode-aware AO minus AO+CEM(46 measurements) over a grid of transmit power x R_min

Protocol (identical to mismatch_study / sweeps): per seed its own layout AND hardware instance, slots at
fixed UAV-to-cluster distances, ONE CSI estimate per slot shared by all methods, paired differences to
nominal-PDA max-min AO per seed, mean +- 95% t-interval over seeds.

Reproducibility: the slot/CSI stream comes from one generator that NO method touches; every method draws
from its own generator seeded by (seed, spread, slot, method name). Adding or removing a method therefore
never changes another method's result, and every curve is paired with ao_nominal on identical slots.

    python -m src.eval.curves calibration --seeds 5 --out results/curves/calibration.json
    python -m src.eval.curves probes      --seeds 5 --out results/curves/probes.json
    python -m src.eval.curves regime      --seeds 5 --out results/curves/regime.json     (resumable)

B counts ALL measurements of the true rate, including the one that scores the starting solution
(cem_ris(include_init=True)); the result is the best MEASURED candidate, so it cannot be worse than the
start on the measured score.
"""
import copy, json, sys, time, zlib
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.ris_baselines import (cascaded_csi, ao_decode, ao_solution, cem_ris, pack_action)  # noqa: E402
from eval.evaluate import ci                                                                      # noqa: E402
from train.train_residual import build_parser, make_slot_env, sample_slot                          # noqa: E402

CAL_ERRORS = [0.0, 0.125, 0.25, 0.5, 1.0, 2.0]
PROBE_BUDGETS = [1, 4, 7, 10, 16, 31, 46, 91]          # B - 1 divisible by the 3 CEM iterations
REGIME_POWERS = [0.0, 5.0, 10.0, 15.0, 20.0]
REGIME_RMINS = [0.25, 0.5, 0.75, 1.0]
REGIME_B = 46


def n_measurements(budget, n_iter=3):
    """True-rate measurements actually spent by cem_ris(budget, include_init=True)."""
    if budget <= 1:
        return 1
    b = budget - 1
    n_iter = max(1, min(n_iter, b))
    return 1 + n_iter * (b // n_iter)


class Slot:
    """One evaluation slot: fixed channel + CSI estimate, per-method generators and a memo for shared work."""

    def __init__(self, env, Dh, seed, spread, idx, a):
        self.env, self.Dh, self.a = env, Dh, a
        self._key = (seed, int(round(100 * spread)), idx)
        self._memo = {}

    def rng(self, name):
        return np.random.default_rng([*self._key, zlib.crc32(name.encode())])

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    # ---- shared solutions ------------------------------------------------------------------------
    def maxmin(self, model):
        return self.memo(('maxmin', model), lambda: ao_solution(
            self.env, self.rng(f'maxmin|{model}'), Dh=self.Dh, model=model, sweeps=self.a.ao_sweeps))

    def decode(self, model):
        return self.memo(('decode', model), lambda: ao_decode(
            self.env, self.rng(f'decode|{model}'), Dh=self.Dh, model=model, sweeps=self.a.ao_sweeps,
            margin=self.a.margin))


def act_maxmin(S, model):
    th, G = S.maxmin(model)
    return pack_action(S.env, G, th)


def act_decode(S, model):
    th, G = S.decode(model)
    return pack_action(S.env, G, th)


def act_cem(S, budget, start):
    """AO+CEM from the 'nominal' max-min or the 'decode' (nominal-model) solution; all measurements counted."""
    th0, _ = S.maxmin('nominal') if start == 'nominal' else S.decode('nominal')
    return cem_ris(S.env, S.rng(f'cem|{start}|{budget}'), budget, model='nominal', Dh=S.Dh,
                   ao_sweeps=S.a.ao_sweeps, init_theta=th0, include_init=True)


def run_seed(a, seed, spread, methods):
    """-> {method: {decode_pct, min_rate}} averaged over the held-out slots of this seed."""
    sa = copy.copy(a); sa.spread = spread
    env = make_slot_env(seed, sa)
    base = np.random.default_rng(10_000 * (seed + 1) + int(100 * spread))   # slot + CSI stream: no method uses it
    acc, idx = {}, 0
    per = max(1, a.study_slots // len(a.eval_dists))
    for d in a.eval_dists:
        for _ in range(per):
            sample_slot(env, base, a.max_dist, fixed_dist=d)
            S = Slot(env, cascaded_csi(env, base, noisy=True), seed, spread, idx, a)
            for name, fn in methods.items():
                r = env.compute_rates_from_action(fn(S))
                acc.setdefault(name, []).append((float(np.mean(r >= env.r_min) * 100), float(r.min())))
            idx += 1
    return {k: dict(decode_pct=float(np.mean([x[0] for x in v])), min_rate=float(np.mean([x[1] for x in v])))
            for k, v in acc.items()}


def summarise(per_seed, keys, seeds, ref='ao_nominal'):
    """per_seed[key][seed][method] -> mean/ci of decode_pct and of the PAIRED difference to `ref`."""
    out = {}
    for key in keys:
        out[key] = {}
        for m in per_seed[key][0]:
            x = [per_seed[key][s][m]['decode_pct'] for s in range(seeds)]
            out[key][m] = dict(decode_pct=dict(zip(('mean', 'ci95'), ci(x))))
            if m != ref:
                d = [per_seed[key][s][m]['decode_pct'] - per_seed[key][s][ref]['decode_pct'] for s in range(seeds)]
                out[key][m]['paired_decode_pct'] = dict(zip(('mean', 'ci95'), ci(d)))
    return out


# --------------------------------------------------------------------------------------------- tasks
def task_calibration(a):
    methods = {'ao_nominal': lambda S: act_maxmin(S, 'nominal'), 'decode_nominal': lambda S: act_decode(S, 'nominal')}
    for e in a.errors:
        methods[f'maxmin_e{e:g}'] = (lambda e: lambda S: act_maxmin(S, f'cal:{e:g}'))(e)
        methods[f'decode_e{e:g}'] = (lambda e: lambda S: act_decode(S, f'cal:{e:g}'))(e)
    return methods


def task_probes(a):
    methods = {'ao_nominal': lambda S: act_maxmin(S, 'nominal'), 'decode_nominal': lambda S: act_decode(S, 'nominal')}
    for B in a.budgets:
        assert n_measurements(B) == B, f'budget {B} is not exactly spendable (use B-1 divisible by 3)'
        methods[f'cem_nominal_B{B}'] = (lambda B: lambda S: act_cem(S, B, 'nominal'))(B)
        methods[f'cem_decode_B{B}'] = (lambda B: lambda S: act_cem(S, B, 'decode'))(B)
    return methods


def task_regime(a):
    return {'ao_nominal': lambda S: act_maxmin(S, 'nominal'), 'decode_nominal': lambda S: act_decode(S, 'nominal'),
            f'cem_nominal_B{REGIME_B}': lambda S: act_cem(S, REGIME_B, 'nominal')}


def main():
    import argparse
    p = build_parser()
    p.add_argument('task', choices=['calibration', 'probes', 'regime'])
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--spreads', type=float, nargs='+', default=None)
    p.add_argument('--study_slots', type=int, default=None)
    p.add_argument('--eval_dists', type=float, nargs='+', default=[0.0, 5.0, 10.0, 15.0])
    p.add_argument('--margin', type=float, default=0.15)
    p.add_argument('--errors', type=float, nargs='+', default=CAL_ERRORS)
    p.add_argument('--budgets', type=int, nargs='+', default=PROBE_BUDGETS)
    p.add_argument('--powers', type=float, nargs='+', default=REGIME_POWERS)
    p.add_argument('--rmins', type=float, nargs='+', default=REGIME_RMINS)
    a = p.parse_args()
    if not a.out:
        p.error('--out is required')
    a.spreads = a.spreads or ([1.0, 2.0] if a.task == 'calibration' else [1.0])
    a.study_slots = a.study_slots or (200 if a.task != 'regime' else 100)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    res['config'] = {k: v for k, v in vars(a).items() if k != 'device'}
    res.setdefault('cells', {})
    t0 = time.time()

    if a.task in ('calibration', 'probes'):
        methods = (task_calibration if a.task == 'calibration' else task_probes)(a)
        cells = [(f'spread={sp:g}', {}, sp) for sp in a.spreads]
    else:
        methods = task_regime(a)
        cells = [(f'power={pw:g}|r_min={rm:g}', dict(power_dbm=pw, r_min=rm), a.spreads[0])
                 for pw in a.powers for rm in a.rmins]

    for key, override, spread in cells:
        if key in res['cells']:
            print(f'skip {key} (done)', flush=True)
            continue
        ca = copy.copy(a)
        for k, v in override.items():
            setattr(ca, k, v)
        per_seed = {key: {}}
        for seed in range(a.seeds):
            per_seed[key][seed] = run_seed(ca, seed, spread, methods)
            print(f'  {key} seed {seed} done ({time.time() - t0:.0f}s)', flush=True)
        res['cells'][key] = dict(overrides=override, spread=spread,
                                 per_seed=[per_seed[key][s] for s in range(a.seeds)],
                                 summary=summarise(per_seed, [key], a.seeds)[key])
        out.write_text(json.dumps(res, indent=2))                  # resumable after every cell

    for key, c in res['cells'].items():
        print(f'\n== {key}')
        for m, v in c['summary'].items():
            pd = v.get('paired_decode_pct')
            print(f"  {m:22s} decode {v['decode_pct']['mean']:6.1f}"
                  + (f"   paired {pd['mean']:+6.1f} +- {pd['ci95']:4.1f}" if pd else ''))
    print('saved', out)


if __name__ == '__main__':
    main()

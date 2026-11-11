"""Statistics + integrity checks for the paper figures (numpy only, no matplotlib/torch).

Every number plotted or tabulated is a mean or a paired mean difference over SEEDS with a two-sided 95% t-interval,
    h = t_{0.975, n-1} * s / sqrt(n),   s = sample std (ddof = 1).
`verify_*` recompute those statistics from the RAW per-seed values and compare them with the summary stored in the
result JSON, so the figures cannot silently disagree with the experiments.
"""
import json
from pathlib import Path

import numpy as np

try:
    from scipy import stats as _st

    def tcrit(df):
        return float(_st.t.ppf(0.975, df))
except Exception:                                    # scipy missing: exact values for small df
    _T = {1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764, 5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060, 9: 2.2622,
          10: 2.2281, 15: 2.1314, 20: 2.0860, 30: 2.0423}

    def tcrit(df):
        if df in _T:
            return _T[df]
        raise ValueError(f'install scipy for df={df}')


def mean_ci(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float('nan')
    return float(x.mean()), float(tcrit(n - 1) * x.std(ddof=1) / np.sqrt(n))


# ----------------------------------------------------------------------------- loading
def load_slot_study(path):
    """-> {spread_str: {method: {metric: np.array over seeds}}} from a mismatch_study/curves-style JSON."""
    d = json.loads(Path(path).read_text())
    out = {}
    for key, v in d['per_seed'].items():
        sp, seed = key.split('|')
        for m, vals in v.items():
            for metric, x in vals.items():
                out.setdefault(sp, {}).setdefault(m, {}).setdefault(metric, {})[int(seed)] = x
    return {sp: {m: {k: np.array([vs[s] for s in sorted(vs)]) for k, vs in ms.items()} for m, ms in mm.items()}
            for sp, mm in out.items()}


def load_sweeps(path):
    """-> {setting: {method: {metric: np.array over seeds}}}"""
    d = json.loads(Path(path).read_text())
    out = {}
    for key, v in d.items():
        out[key] = {}
        for seed_res in v['per_seed']:
            for m, vals in seed_res.items():
                for metric, x in vals.items():
                    out[key].setdefault(m, {}).setdefault(metric, []).append(x)
        out[key] = {m: {k: np.array(x) for k, x in ms.items()} for m, ms in out[key].items()}
    return out


def load_aoi(path):
    """-> {'sched+lowlevel': {metric: np.array over seeds}}"""
    d = json.loads(Path(path).read_text())['per_seed']
    out = {}
    for key, vals in d.items():
        name, seed = key.rsplit('|', 1)
        for metric, x in vals.items():
            out.setdefault(name, {}).setdefault(metric, {})[int(seed)] = x
    return {n: {k: np.array([vs[s] for s in sorted(vs)]) for k, vs in ms.items()} for n, ms in out.items()}


# ----------------------------------------------------------------------------- verification
def _close(a, b, rtol, atol, what):
    if not np.isclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f'{what}: stored {b!r} != recomputed {a!r}')


def verify_slot_study(path, ref='ao_nominal'):
    """Recompute mean, CI and paired-vs-ref statistics from per_seed and compare with the stored summary."""
    d = json.loads(Path(path).read_text())
    raw, n = load_slot_study(path), 0
    for sp, methods in d['summary'].items():
        for m, st in methods.items():
            for metric, v in st.items():
                if metric.startswith('paired_'):
                    base = metric[len('paired_'):]
                    x = raw[sp][m][base] - raw[sp][ref][base]
                elif metric == 'probes_per_slot':
                    continue
                else:
                    x = raw[sp][m][metric]
                mean, h = mean_ci(x)
                _close(mean, v['mean'], 1e-9, 1e-9, f'{path.name if hasattr(path, "name") else path} s={sp} {m} {metric} mean')
                _close(h, v['ci95'], 5e-3, 1e-6, f'{path} s={sp} {m} {metric} ci95')   # stored CIs used a 4-digit t table
                n += 2
            dp = raw[sp][m]['decode_pct']
            assert np.all((dp >= 0) & (dp <= 100)), f'decode_pct outside [0,100]: s={sp} {m}'
    return n


def verify_sweeps(path, ref='ao_nominal'):
    d = json.loads(Path(path).read_text())
    raw, n = load_sweeps(path), 0
    for key, v in d.items():
        for m, st in v['summary'].items():
            for metric, s in st.items():
                if metric.startswith('paired_'):
                    x = raw[key][m][metric[len('paired_'):]] - raw[key][ref][metric[len('paired_'):]]
                else:
                    x = raw[key][m][metric]
                mean, h = mean_ci(x)
                _close(mean, s['mean'], 1e-9, 1e-9, f'sweeps {key} {m} {metric} mean')
                _close(h, s['ci95'], 5e-3, 1e-6, f'sweeps {key} {m} {metric} ci95')
                n += 2
    return n


def verify_aoi(path):
    d = json.loads(Path(path).read_text())
    raw, n = load_aoi(path), 0
    for name, st in d['summary'].items():
        for metric, s in st.items():
            mean, h = mean_ci(raw[name][metric])
            _close(mean, s['mean'], 1e-9, 1e-9, f'aoi {name} {metric} mean')
            _close(h, s['ci95'], 5e-3, 1e-6, f'aoi {name} {metric} ci95')
            n += 2
    return n


def verify_all(data_dir):
    """Check every result file that exists in paper/data; returns (n_statistics, list of files checked)."""
    data_dir = Path(data_dir)
    total, files = 0, []
    jobs = [(verify_slot_study, data_dir / '03_headroom' / 'headroom_v5_FINAL_code_d66def0.json'),
            (verify_sweeps, data_dir / '07_sweeps' / 'sweeps.json'),
            (verify_aoi, data_dir / '05_macro_and_aoi' / 'aoi_v5.json'),
            (verify_aoi, data_dir / '05_macro_and_aoi' / 'aoi_v5_spread2.json'),
            (verify_slot_study, data_dir / '08_curves' / 'curves_main.json'),
            (verify_slot_study, data_dir / '08_curves' / 'curves_spread.json')]
    for fn, p in jobs:
        if p.exists():
            total += fn(p)
            files.append(p.name)
    return total, files

#!/usr/bin/env python3
"""Turn result JSONs into paper-ready figures (PDF) and LaTeX tables. Missing inputs are skipped.

    python -m src.eval.make_report --out results/report
"""
import argparse, glob, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'serif', 'font.size': 10, 'axes.grid': True, 'grid.alpha': 0.3,
                     'savefig.bbox': 'tight', 'figure.dpi': 150})
NAMES = {'ao_nominal': 'AO (nominal PDA)', 'ao_ideal': 'AO (ideal RIS assumed)', 'ao_calibrated': 'AO (true PDA curve, noisy CSI)',
         'ao_nominal_perfcsi': 'AO (nominal PDA, perfect CSI)', 'ao_calibrated_perfcsi': 'AO (true PDA, perfect CSI)',
         'cem6': 'AO+CEM, 6 probes', 'cem15': 'AO+CEM, 15 probes', 'cem45': 'AO+CEM, 45 probes',
         'best_of_45': 'Random search, 45 probes', 'residual_sac': 'Physics-guided residual SAC (proposed)'}
ORDER = ['ao_ideal', 'ao_nominal', 'best_of_45', 'cem6', 'cem15', 'cem45', 'residual_sac', 'ao_calibrated',
         'ao_nominal_perfcsi', 'ao_calibrated_perfcsi']


def mismatch(path, out):
    d = json.loads(Path(path).read_text())['summary']
    spreads = sorted(d, key=float)
    methods = [m for m in ORDER if m in d[spreads[0]]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, lab in zip(axes, ('decode_pct', 'min_rate'), ('Decode rate (%)', 'Min-user rate (bit/s/Hz)')):
        for m in methods:
            y = [d[s][m][metric]['mean'] for s in spreads]; e = [d[s][m][metric]['ci95'] for s in spreads]
            ax.errorbar([float(s) for s in spreads], y, yerr=e, marker='o', ms=4, capsize=2, label=NAMES.get(m, m),
                        lw=2.2 if m == 'residual_sac' else 1.2)
        ax.set_xlabel('Hardware mismatch level'); ax.set_ylabel(lab)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.savefig(out / 'fig_mismatch.pdf'); plt.close(fig)

    rows = ['\\begin{tabular}{l' + 'rr' * len(spreads) + '}', '\\toprule',
            'Method & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{spread {s}}}' for s in spreads) + ' \\\\',
            ' & ' + ' & '.join('Dec.\\,\\% & Min-rate' for _ in spreads) + ' \\\\', '\\midrule']
    for m in methods:
        cells = []
        for s in spreads:
            v = d[s][m]
            cells += [f"{v['decode_pct']['mean']:.1f}$\\pm${v['decode_pct']['ci95']:.1f}",
                      f"{v['min_rate']['mean']:.2f}$\\pm${v['min_rate']['ci95']:.2f}"]
        rows.append(NAMES.get(m, m) + ' & ' + ' & '.join(cells) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    (out / 'table_mismatch.tex').write_text('\n'.join(rows))


def residual_curves(root, out):
    files = sorted(glob.glob(str(Path(root) / 'seed*_spread*' / 'history.json')))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    gains = {}
    for f in files:
        h = json.loads(Path(f).read_text()); spread = Path(f).parent.name.split('spread')[1]
        gains.setdefault(spread, []).append(([x['slots'] for x in h], [x['policy_reward'] - x['prior_reward'] for x in h]))
    for spread, runs in gains.items():
        n = min(len(r[0]) for r in runs)
        xs = runs[0][0][:n]; ys = np.array([r[1][:n] for r in runs])
        ax.plot(xs, ys.mean(0), label=f'spread {spread}')
        ax.fill_between(xs, ys.mean(0) - ys.std(0), ys.mean(0) + ys.std(0), alpha=0.2)
    ax.axhline(0, color='k', lw=0.8); ax.set_xlabel('Training slots'); ax.set_ylabel('Reward gain over AO prior')
    ax.legend(); fig.savefig(out / 'fig_residual_learning.pdf'); plt.close(fig)


def aoi_table(path, out):
    d = json.loads(Path(path).read_text())['summary']
    rows = ['\\begin{tabular}{lrrr}', '\\toprule', 'Scheduler + RIS policy & Avg AoI & Peak AoI & Decode \\% \\\\', '\\midrule']
    for k, v in d.items():
        name = k.replace('_', '\\_')
        cells = [f"{v[m]['mean']:.1f}$\\pm${v[m]['ci95']:.1f}" for m in ('avg_aoi', 'peak_aoi', 'decode_pct')]
        rows.append(name + ' & ' + ' & '.join(cells) + ' \\\\')
    rows += ['\\bottomrule', '\\end{tabular}']
    (out / 'table_aoi.tex').write_text('\n'.join(rows))


def macro_curve(root, out):
    files = sorted(glob.glob(str(Path(root) / '*' / 'log.json')))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for f in files:
        h = json.loads(Path(f).read_text()); x = [r['episode'] for r in h]
        ax.plot(x, [r['ppo_eval_aoi'] for r in h], label=f'PPO ({Path(f).parent.name})')
        ax.plot(x, [r['greedy_eval_aoi'] for r in h], '--', label=f'greedy ({Path(f).parent.name})')
    ax.set_xlabel('Training episode'); ax.set_ylabel('Held-out average AoI'); ax.legend(fontsize=7)
    fig.savefig(out / 'fig_macro_learning.pdf'); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='results/report')
    p.add_argument('--results', default='results')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); R = Path(a.results)
    if (R / 'mismatch_study/mismatch_study.json').exists():
        mismatch(R / 'mismatch_study/mismatch_study.json', out)
    residual_curves(R / 'residual', out)
    if (R / 'aoi_study/aoi_study.json').exists():
        aoi_table(R / 'aoi_study/aoi_study.json', out)
    macro_curve(R / 'macro', out)
    print('report written to', out)


if __name__ == '__main__':
    main()

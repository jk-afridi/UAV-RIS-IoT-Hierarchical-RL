#!/usr/bin/env python3
"""Generate the paper figures from the curated JSON in paper/data (no torch).   python paper/make_figures.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / 'data', ROOT / 'figures'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
                     'savefig.bbox': 'tight', 'figure.dpi': 150})
C = {'a': '#0072B2', 'b': '#E69F00', 'c': '#009E73', 'd': '#CC79A7', 'e': '#56B4E9', 'f': '#D55E00'}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def fig_headroom():
    s = json.loads((DATA / '03_headroom' / 'headroom_v5_FINAL_code_d66def0.json').read_text())['summary']
    spreads = list(s)
    methods = [('ao_calibrated', 'AO, true PDA', C['a']), ('ao_nominal_perfcsi', 'AO, perfect CSI', C['e']),
               ('ao_decode_nominal', 'Decode-aware AO', C['c']), ('ao_decode_calibrated', 'Decode-aware AO, true PDA', C['d']),
               ('cem15', 'CEM, 15 probes', C['b']), ('cem45', 'CEM, 45 probes', C['f'])]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    w = 0.13
    for i, (m, lab, col) in enumerate(methods):
        y = [s[sp][m]['paired_decode_pct']['mean'] for sp in spreads]
        e = [s[sp][m]['paired_decode_pct']['ci95'] for sp in spreads]
        ax.bar(np.arange(len(spreads)) + (i - 2.5) * w, y, w, yerr=e, capsize=1.5, color=col, label=lab, error_kw={'lw': 0.7})
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(len(spreads))); ax.set_xticklabels([f'$s={float(x):g}$' for x in spreads])
    ax.set_ylabel('Decode-rate gain vs nominal AO (pp)'); ax.legend(fontsize=5.5, ncol=2, loc='upper left')
    fig.savefig(OUT / 'fig_headroom.pdf'); plt.close(fig)


def fig_sweeps():
    d = json.loads((DATA / '07_sweeps' / 'sweeps.json').read_text())
    keys = list(d)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    y = np.arange(len(keys))[::-1]
    for off, (m, lab, col) in zip((-0.2, 0.0, 0.2), [('ao_decode_nominal', 'Decode-aware AO (0 probes)', C['c']),
                                                    ('cem15', 'CEM, 15 probes', C['b']), ('cem45', 'CEM, 45 probes', C['f'])]):
        mu = [d[k]['summary'][m]['paired_decode_pct']['mean'] for k in keys]
        e = [d[k]['summary'][m]['paired_decode_pct']['ci95'] for k in keys]
        ax.errorbar(mu, y + off, xerr=e, fmt='o', ms=3, lw=0.8, capsize=1.5, color=col, label=lab)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=6)
    ax.set_xlabel('Decode-rate gain vs nominal AO (pp)'); ax.legend(fontsize=6, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3, columnspacing=0.8, handletextpad=0.3)
    fig.savefig(OUT / 'fig_sweeps.pdf'); plt.close(fig)


def fig_aoi():
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.2), sharey=True)
    names = [('ao_decode_nominal', 'Decode-aware', C['c']), ('ao_decode_calibrated', 'Decode-aware, true PDA', C['d']),
             ('ao_calibrated', 'AO, true PDA', C['a'])]
    for ax, (f, ttl) in zip(axes, (('aoi_v5.json', '$s=1$'), ('aoi_v5_spread2.json', '$s=2$'))):
        ps = json.loads((DATA / '05_macro_and_aoi' / f).read_text())['per_seed']
        for j, sc in enumerate(('greedy_aoi_travel', 'greedy_aoi')):
            for i, (ll, lab, col) in enumerate(names):
                x = np.array([ps[f'{sc}+{ll}|{s}']['avg_aoi'] - ps[f'{sc}+ao_nominal|{s}']['avg_aoi'] for s in range(5)])
                h = T975[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x))
                ax.bar(j + (i - 1) * 0.26, x.mean(), 0.26, yerr=h, capsize=1.5, color=col, label=lab if j == 0 else None, error_kw={'lw': 0.7})
        ax.axhline(0, color='k', lw=0.8); ax.set_title(ttl, fontsize=8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['greedy\n+travel', 'greedy'], fontsize=6.5)
    axes[0].set_ylabel(r'$\Delta$ avg. AoI vs nominal AO (slots)'); axes[0].legend(fontsize=5.5, loc='lower left')
    fig.savefig(OUT / 'fig_aoi.pdf'); plt.close(fig)


if __name__ == '__main__':
    fig_headroom(); fig_sweeps(); fig_aoi()
    print('wrote', *sorted(p.name for p in OUT.glob('fig_*.pdf')))

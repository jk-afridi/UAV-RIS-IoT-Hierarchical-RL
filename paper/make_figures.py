#!/usr/bin/env python3
"""
Paper figures, generated ONLY from the stored per-seed experiment data, with built-in verification.

    python paper/make_figures.py --curves results/curves --out results/paper_figures      # Colab
    python paper/make_figures.py                                                          # from paper/data

Every plotted statistic is RECOMPUTED here from per-seed raw values (mean and 95% t-interval, paired
differences to nominal-PDA max-min AO) and compared with the summary that the experiment script stored;
mathematical invariants (PDA curve extrema, hardware-generator moments, budget-1 CEM == its start, ...)
are asserted. Results go to <out>/figure_verification.json; all plotted numbers go to <out>/csv/*.csv.
The process exits non-zero if a hard check fails, but figures are always written first.

Figures (IEEE column widths: 3.5 in, 7.16 in):
  fig_pda_model.pdf       true per-element PDA curves vs the nominal curve at mismatch levels s
  fig_headroom.pdf        forest plot of paired decode-rate gains at s = 0, 1, 2 with per-seed points
  fig_calibration.pdf     gain vs calibration error e (max-min AO and decode-aware AO)          [curves]
  fig_probes.pdf          gain vs number of true-rate measurements B, with break-even budget     [curves]
  fig_regime.pdf          decode-aware AO vs AO+CEM(46) over transmit power x R_min              [curves]
  fig_episode.pdf         episode-level paired change in AoI and decode rate, per seed
  fig_residual.pdf        learned-residual SAC gain over its AO prior during training
  fig_power.pdf           headline gain with 20 seeds: exploratory (0-4) vs confirmatory (5-19), running interval   [curves]
  fig_robust.pdf          gain vs RIS phase resolution and vs CSI-error power                                  [curves]
  fig_tau.pdf             decode-aware gain over the (sigmoid width x margin) grid                              [curves]
  fig_width.pdf           sigmoid width: (a) grid on design seeds, (b,c) width chosen on seeds 0-4, reported on seeds 5-19 [curves]
  fig_probing.pdf         probing budget + regime maps in one wide figure (replaces fig_probes_tuned / fig_regime_tuned in the paper) [curves]
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy import stats

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / 'src'))

# Reference categorical palette of the dataviz skill (validated for adjacency CVD/contrast); marker shapes
# and line styles carry the same identity, so nothing relies on colour alone (print / greyscale safe).
BLUE, ORANGE, AQUA, VIOLET, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#6b6b68', '#0b0b0b', '#52514e'
GROUP = {'model': (BLUE, 'o', 'Model-based, no probes'), 'decode': (AQUA, 's', 'Decode-aware AO, no probes'),
         'probe': (ORANGE, 'D', 'Probing (true-rate feedback)'), 'diag': (VIOLET, '^', 'Diagnostic (perfect CSI)')}

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 7.5, 'axes.labelsize': 7.5,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
                     'grid.linewidth': 0.4, 'grid.alpha': 0.35, 'axes.grid': True, 'axes.axisbelow': True,
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.dpi': 150})

CHECKS = []          # every verification, written to figure_verification.json


# ------------------------------------------------------------------------------------------ statistics
def tci(x):
    """mean and 95% half-width of a t-interval (scipy, exact quantile)."""
    x = np.asarray(x, float)
    n = len(x)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def paired(per_seed, method, ref, metric='decode_pct'):
    """per-seed paired differences method - ref; per_seed: list over seeds of {method: {metric: v}}."""
    return np.array([s[method][metric] - s[ref][metric] for s in per_seed], float)


def check(name, got, ref, hard=True, rtol=2e-3, atol=1e-6):
    """Compare a recomputed value with a stored one (the stored CI uses a 3-digit t table, hence rtol)."""
    got, ref = float(got), float(ref)
    ok = abs(got - ref) <= atol + rtol * abs(ref)
    CHECKS.append(dict(name=name, recomputed=got, stored=ref, abs_err=abs(got - ref), ok=bool(ok), hard=hard))
    return ok


def invariant(name, cond, detail='', hard=True):
    CHECKS.append(dict(name=name, ok=bool(cond), detail=str(detail), hard=hard))
    return bool(cond)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight', pad_inches=0.03)
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('  wrote', name)


def load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


# ------------------------------------------------------------------------------------------ Fig: seed power
N_EXPLORATORY = 5      # seeds 0-4 were used while the method was designed; seeds 5.. were never used for any choice


def fig_power(curves, out):
    """Headline paired gain (decode-aware vs nominal AO) with many seeds: per-seed values, running 95% interval,
    and the pre-declared confirmatory test on the seeds that played no part in designing the method."""
    d = load(curves / 'power.json')
    if d is None:
        return print('  skip fig_power (power.json missing)')
    cells = list(d['cells'])
    fig, axes = plt.subplots(2, len(cells), figsize=(3.5 if len(cells) == 1 else 7.16, 4.0), squeeze=False, constrained_layout=True)
    rows_csv, summary = [], {}
    for j, key in enumerate(cells):
        c = d['cells'][key]; ps = c['per_seed']; n = len(ps); sp = c['spread']
        for fam, lab, col, mk in (('decode_nominal', 'decode-aware AO, nominal model', AQUA, 's'),):
            x = paired(ps, fam, 'ao_nominal')
            conf = x[N_EXPLORATORY:]
            m_all, h_all = tci(x); m_c, h_c = tci(conf); m_e, h_e = tci(x[:N_EXPLORATORY])
            st = c['summary'][fam]['paired_decode_pct']
            check(f'power {key} all-seed mean', m_all, st['mean'])
            # the stored CI of this run used a t table without df=19 (fell back to the next lower df = a LARGER multiplier):
            # the stored value must be the exact one or a conservative (wider) one, never narrower, and within that table's step
            invariant(f'power {key} all-seed CI: stored interval is exact or conservative (never narrower)',
                      h_all * (1 - 2e-3) <= st['ci95'] <= h_all * stats.t.ppf(0.975, 4) / stats.t.ppf(0.975, n - 1) * 1.001,
                      f'exact {h_all:.4f}, stored {st["ci95"]:.4f}')
            k = int((conf > 0).sum())
            p_sign = float(stats.binomtest(k, len(conf), 0.5, alternative='greater').pvalue)
            # declared criteria (written before the run): confirmatory interval excludes 0 AND >= 90% of confirmatory seeds improve
            invariant(f'power {key}: confirmatory 95% interval excludes 0', m_c - h_c > 0, f'{m_c:.2f} +- {h_c:.2f} (n={len(conf)})', hard=False)
            invariant(f'power {key}: >= 90% of confirmatory seeds improve', k / len(conf) >= 0.9, f'{k}/{len(conf)}', hard=False)
            summary[key] = dict(n=n, n_conf=len(conf), mean_all=m_all, ci_all=h_all, mean_conf=m_c, ci_conf=h_c,
                                mean_expl=m_e, ci_expl=h_e, k_pos_conf=k, sign_test_p=p_sign)
            rows_csv.append([key, 'per_seed'] + list(x))
            ax = axes[0][j]
            idx = np.arange(n)
            ax.axvspan(-0.5, N_EXPLORATORY - 0.5, color=GREY, alpha=0.10, lw=0)
            ax.plot(idx[:N_EXPLORATORY], x[:N_EXPLORATORY], mk, ms=4.2, mfc='white', mec=col, mew=1.0, ls='', label='exploratory seeds (0-4)')
            ax.plot(idx[N_EXPLORATORY:], conf, mk, ms=4.2, mfc=col, mec=INK, mew=0.4, ls='', label=f'confirmatory seeds ({N_EXPLORATORY}-{n - 1})')
            ax.axhspan(m_c - h_c, m_c + h_c, xmin=0.0, xmax=1.0, color=col, alpha=0.14, lw=0)
            ax.axhline(m_c, color=col, lw=0.9)
            ax.axhline(0, color=INK, lw=0.7)
            ax.set_title(f'$s={sp:g}$:  confirmatory mean ${m_c:+.1f}\\pm{h_c:.1f}$ pp, {k}/{len(conf)} seeds improve', fontsize=6.8)
            ax.set_xlabel('seed'); ax.set_xlim(-0.7, n - 0.3)
            # running interval as seeds accumulate (analytic t quantile each time)
            ax2 = axes[1][j]
            ns = np.arange(2, n + 1)
            run = np.array([tci(x[:q]) for q in ns])
            ax2.fill_between(ns, run[:, 0] - run[:, 1], run[:, 0] + run[:, 1], color=col, alpha=0.18, lw=0)
            ax2.plot(ns, run[:, 0], '-', color=col, lw=1.1)
            ax2.axhline(0, color=INK, lw=0.7)
            ax2.axvline(N_EXPLORATORY, color=GREY, lw=0.7, ls=':')
            ax2.set_xlabel('number of seeds used'); ax2.set_xlim(2, n)
            invariant(f'power {key}: running interval at n={n} equals the full-sample interval', abs(run[-1, 1] - h_all) < 1e-9)
            rows_csv.append([key, 'running_mean'] + list(run[:, 0])); rows_csv.append([key, 'running_ci95'] + list(run[:, 1]))
    axes[0][0].set_ylabel('paired gain vs nominal AO (pp)'); axes[1][0].set_ylabel('mean and 95% $t$-interval (pp)')
    h_, l_ = axes[0][0].get_legend_handles_labels()
    fig.legend(h_, l_, loc='outside lower center', ncol=2, frameon=False)
    save(fig, out, 'fig_power')
    (out / 'csv').mkdir(parents=True, exist_ok=True)
    (out / 'csv' / 'fig_power_summary.json').write_text(json.dumps(summary, indent=1))
    write_csv(out / 'csv' / 'fig_power.csv', ['cell', 'series'] + [f'v{i}' for i in range(max(len(r) for r in rows_csv) - 2)], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: hardware realism
def fig_robust(curves, out):
    """(a) phase resolution, (b) channel-estimation error. Both reuse the paired protocol."""
    dq, dc = load(curves / 'quant.json'), load(curves / 'csi.json')
    if dq is None and dc is None:
        return print('  skip fig_robust (quant.json and csi.json missing)')
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), constrained_layout=True)
    rows_csv = []
    if dq is not None:
        ax = axes[0]; key = list(dq['cells'])[0]; c = dq['cells'][key]; ps = c['per_seed']; bits = dq['config']['bits']
        for fam, lab, col, mk, ls in (('maxmin', 'max-min AO', BLUE, 'o', '-'), ('decode', 'decode-aware AO', AQUA, 's', '--')):
            mu, hw = [], []
            for b in bits:
                x = paired(ps, f'{fam}_b{b}', 'ao_nominal'); m, h = tci(x)
                st = c['summary'][f'{fam}_b{b}']['paired_decode_pct']
                check(f'quant {fam} b={b} mean', m, st['mean']); check(f'quant {fam} b={b} CI', h, st['ci95'])
                mu.append(m); hw.append(h); rows_csv.append(['quant', fam, b, m, h] + list(x))
            ax.errorbar(bits, mu, yerr=hw, fmt=mk + ls, ms=3.8, color=col, mec=INK, mew=0.4, elinewidth=0.9, capsize=2, lw=1.1, label=lab)
        if 6 in bits:      # 6 bits is the unrestricted 64-point grid: it must reproduce the reference solvers EXACTLY
            dm = max(abs(s['maxmin_b6']['decode_pct'] - s['ao_nominal']['decode_pct']) for s in ps)
            dd = max(abs(s['decode_b6']['decode_pct'] - s['decode_nominal']['decode_pct']) for s in ps)
            invariant('quant: 6-bit max-min AO equals the unrestricted nominal AO', dm < 1e-9, f'max |diff|={dm:.3g} pp')
            invariant('quant: 6-bit decode-aware AO equals the unrestricted decode-aware AO', dd < 1e-9, f'max |diff|={dd:.3g} pp')
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xlabel('RIS phase resolution (bits; 6 = unrestricted grid)'); ax.set_ylabel('decode-rate gain vs 6-bit nominal AO (pp)')
        ax.set_xticks(bits); ax.set_title('(a) phase quantisation, $s=1$', fontsize=7.2); ax.legend(frameon=False, loc='lower right')
    if dc is not None:
        ax = axes[1]; keys = list(dc['cells']); B = dc['config'].get('budgets', [46])
        x_db = np.array([np.mean([s['_meta']['csi_nmse_db'] for s in dc['cells'][k]['per_seed']]) for k in keys])
        invariant('csi: error power (dB) increases with the configured variance', bool(np.all(np.diff(x_db) > 0)), str(np.round(x_db, 1).tolist()))
        for m_, lab, col, mk, ls in (('decode_nominal', 'decode-aware AO', AQUA, 's', '--'), ('perfect_maxmin', 'max-min AO, perfect CSI', VIOLET, '^', ':'),
                                     ('cem_nominal_B46', 'AO+CEM, 46 probes', ORANGE, 'D', '-')):
            mu, hw = [], []
            for k in keys:
                ps = dc['cells'][k]['per_seed']; x = paired(ps, m_, 'ao_nominal'); m, h = tci(x)
                st = dc['cells'][k]['summary'][m_]['paired_decode_pct']
                check(f'csi {m_} {k} mean', m, st['mean']); check(f'csi {m_} {k} CI', h, st['ci95'])
                mu.append(m); hw.append(h); rows_csv.append(['csi', m_, k, m, h] + list(x))
            ax.errorbar(x_db, mu, yerr=hw, fmt=mk + ls, ms=3.8, color=col, mec=INK, mew=0.4, elinewidth=0.9, capsize=2, lw=1.1, label=lab)
        # exact invariant: solvers that see the clean channel cannot depend on the CSI error level
        pd_ = [[s['perfect_decode']['decode_pct'] for s in dc['cells'][k]['per_seed']] for k in keys]
        invariant('csi: perfect-CSI decode-aware result is identical across CSI-error levels', all(np.allclose(pd_[0], r, atol=1e-9) for r in pd_))
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xlabel('CSI error power relative to channel power (dB)'); ax.set_ylabel('decode-rate gain vs nominal AO (pp)')
        ax.set_title('(b) channel-estimation error, $s=1$', fontsize=7.2); ax.legend(frameon=False, loc='best')
    save(fig, out, 'fig_robust')
    write_csv(out / 'csv' / 'fig_robust.csv', ['panel', 'method', 'x', 'mean_pp', 'ci95_pp'] + [f'seed{i}' for i in range(5)], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: objective sensitivity
def fig_tau(curves, out, ax=None, fig=None, title=None):
    d = load(curves / 'tau.json')
    if d is None:
        return print('  skip fig_tau (tau.json missing)')
    key = list(d['cells'])[0]; c = d['cells'][key]; ps = c['per_seed']
    taus, margins = d['config']['taus'], d['config']['margins']
    G, H = np.full((len(margins), len(taus)), np.nan), np.full((len(margins), len(taus)), np.nan)
    rows_csv = []
    for i, mg in enumerate(margins):
        for j, t in enumerate(taus):
            x = paired(ps, f'tm_t{t:g}_m{mg:g}', 'ao_nominal'); m, h = tci(x)
            st = c['summary'][f'tm_t{t:g}_m{mg:g}']['paired_decode_pct']
            check(f'tau t={t:g} m={mg:g} mean', m, st['mean']); check(f'tau t={t:g} m={mg:g} CI', h, st['ci95'])
            G[i, j], H[i, j] = m, h; rows_csv.append([t, mg, m, h] + list(x))
    n_pos = int((G > 0).sum()); n_cert = int((G - H > 0).sum())
    invariant('tau grid: decode-aware AO improves on max-min AO in every (tau, margin) cell (declared criterion)', n_pos == G.size, f'{n_pos}/{G.size} cells with positive mean', hard=False)
    CHECKS.append(dict(name='tau grid: cells whose 95% interval excludes 0', value=n_cert, of=int(G.size), ok=True, hard=False))
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
    cm = LinearSegmentedColormap.from_list('seq', ['#f3f7fc', BLUE])
    im = ax.imshow(G, origin='lower', cmap=cm, aspect='auto', vmin=min(0, np.nanmin(G)))
    ax.set_xticks(range(len(taus))); ax.set_xticklabels([f'{t:g}' for t in taus])
    ax.set_yticks(range(len(margins))); ax.set_yticklabels([f'{m:g}' for m in margins])
    for i in range(len(margins)):
        for j in range(len(taus)):
            ax.text(j, i, f'{G[i, j]:+.1f}' + ('' if G[i, j] - H[i, j] > 0 else '$^\\dagger$'), ha='center', va='center', fontsize=6.2, color=INK)
    if 0.2 in taus and 0.15 in margins:
        jj, ii = taus.index(0.2), margins.index(0.15)
        ax.add_patch(plt.Rectangle((jj - 0.5, ii - 0.5), 1, 1, fill=False, ec=INK, lw=1.3))
    ax.grid(False)
    ax.set_xlabel(r'sigmoid width $\tau / R_{\min}$'); ax.set_ylabel(r'margin $\epsilon$')
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, label='gain vs nominal AO (pp)')
    if title:
        ax.set_title(title, fontsize=7.2)
    if standalone:
        fig.text(0.5, -0.02, 'Box: setting used in the paper (5 seeds).' + (r'  $\dagger$: 95% interval includes 0.' if n_cert < G.size else '  All intervals exclude 0.'),
                 ha='center', va='top', fontsize=5.8, color=MUTED)
        save(fig, out, 'fig_tau')
    write_csv(out / 'csv' / 'fig_tau.csv', ['tau_frac', 'margin', 'mean_pp', 'ci95_pp'] + [f'seed{s}' for s in range(len(ps))], rows_csv)


def fig_tau_confirm(curves, out, axs=None):
    """Sigmoid width chosen on the exploratory seeds (0-4, s=1) only, then reported on the 15 confirmatory seeds."""
    d = load(curves / 'tau_confirm.json')
    if d is None:
        return print('  skip fig_tau_confirm (tau_confirm.json missing)')
    taus, mg = d['config']['taus'], d['config']['margin']
    cells = list(d['cells']); ref_tau = 0.2
    nm = lambda t: f'tm_t{t:g}_m{mg:g}'
    c1 = d['cells'][cells[0]]
    sel = [tci(paired(c1['per_seed'][:N_EXPLORATORY], nm(t), 'ao_nominal'))[0] for t in taus]
    t_star = taus[int(np.argmax(sel))]                      # selection uses ONLY seeds 0-4 of the first cell (s=1)
    standalone = axs is None
    if standalone:
        fig, axes = plt.subplots(1, len(cells), figsize=(7.16, 2.5), sharey=True, constrained_layout=True, squeeze=False)
        axs = list(axes[0])
    rows_csv, summary = [], dict(tau_star=t_star, selected_on='seeds 0-4, s=1')
    for ax, key in zip(axs, cells):
        c = d['cells'][key]; ps = c['per_seed']; sp = c['spread']
        for lab, sl, filled in (('exploratory seeds (0-4)', slice(0, N_EXPLORATORY), False), ('confirmatory seeds (5-19)', slice(N_EXPLORATORY, None), True)):
            mu, hw = [], []
            for t in taus:
                x = paired(ps[sl], nm(t), 'ao_nominal'); m, h = tci(x); mu.append(m); hw.append(h)
                rows_csv.append([key, lab, t, m, h])
            ax.errorbar(np.array(taus) * (0.97 if not filled else 1.03), mu, yerr=hw, fmt='s' + ('-' if filled else '--'), ms=3.8, color=AQUA,
                        mfc=AQUA if filled else 'white', mec=INK if filled else AQUA, mew=0.6, elinewidth=0.9, capsize=2, lw=1.0 if filled else 0.8, label=lab)
        full = tci(paired(ps, nm(t_star), 'ao_nominal')); st = c['summary'][nm(t_star)]['paired_decode_pct']
        check(f'tau_confirm {key} tau*={t_star:g} all-seed mean', full[0], st['mean'])
        conf = paired(ps[N_EXPLORATORY:], nm(t_star), 'ao_nominal'); m_c, h_c = tci(conf)
        gain_over_default = paired(ps[N_EXPLORATORY:], nm(t_star), nm(ref_tau)); m_d, h_d = tci(gain_over_default)
        k = int((conf > 0).sum())
        invariant(f'tau_confirm {key}: confirmatory interval of tau*={t_star:g} excludes 0', m_c - h_c > 0, f'{m_c:.2f} +- {h_c:.2f}', hard=False)
        invariant(f'tau_confirm {key}: tau* beats the default tau=0.2 on the confirmatory seeds (paired interval excludes 0)', m_d - h_d > 0, f'{m_d:.2f} +- {h_d:.2f}', hard=False)
        summary[key] = dict(mean_conf=m_c, ci_conf=h_c, k_pos=k, n_conf=len(conf), vs_default_mean=m_d, vs_default_ci=h_d)
        ax.axvline(t_star, color=GREY, lw=0.7, ls=':'); ax.axhline(0, color=INK, lw=0.7)
        ax.set_xscale('log'); ax.set_xticks(taus); ax.set_xticklabels([f'{t:g}' for t in taus]); ax.minorticks_off()
        ax.set_xlabel(r'sigmoid width $\tau / R_{\min}$'); ax.set_title(f'$s={sp:g}$: ${m_c:+.1f}\\pm{h_c:.1f}$ pp at $\\tau^\\star$ ({k}/{len(conf)} seeds)', fontsize=6.6)
    axs[0].set_ylabel('gain vs nominal max-min AO (pp)'); axs[0].legend(frameon=False, loc='upper left', fontsize=6)
    if standalone:
        save(fig, out, 'fig_tau_confirm')
    write_csv(out / 'csv' / 'fig_tau_confirm.csv', ['cell', 'seeds', 'tau_frac', 'mean_pp', 'ci95_pp'], rows_csv)
    (out / 'csv' / 'fig_tau_confirm_summary.json').write_text(json.dumps(summary, indent=1))


def fig_width(curves, out):
    """One wide figure for the sigmoid-width question: (a) grid on the design seeds, (b),(c) held-out confirmation."""
    if load(curves / 'tau.json') is None or load(curves / 'tau_confirm.json') is None:
        return print('  skip fig_width (tau.json / tau_confirm.json missing)')
    fig = plt.figure(figsize=(7.16, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.05])
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2], sharey=ax1)
    fig_tau(curves, out, ax=ax0, fig=fig, title='(a) width x margin, design seeds')
    fig_tau_confirm(curves, out, axs=[ax1, ax2])
    ax1.set_title('(b) ' + ax1.get_title(), fontsize=6.8); ax2.set_title('(c) ' + ax2.get_title(), fontsize=6.8)
    ax2.tick_params(labelleft=False)
    save(fig, out, 'fig_width')


# ------------------------------------------------------------------------------------------ Fig: PDA model
def fig_pda_model(out):
    from baselines.ris_baselines import _amp_theta
    from core.aoi_environment import AoIUAVRISEnvironment

    b0, k0, t0 = 0.2, 1.5, 0.0
    theta = np.linspace(-np.pi, np.pi, 721)
    closed = lambda th, b, k, t: (1 - b) * ((np.sin(th - t) + 1) / 2) ** k + b     # Eq. (2) of the paper

    spreads = [0.5, 1.0, 2.0]
    pooled = {s: dict(th=[], lk=[], b=[]) for s in spreads}
    envs = {}
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), sharey=True)
    rows = []
    for ax, s in zip(axes, spreads):
        for seed in range(5):
            if seed not in envs:                           # same construction as train_residual.make_slot_env
                envs[seed] = AoIUAVRISEnvironment(num_antennas=4, num_RIS_elements=64, num_users=4, beta_min=b0, kappa_bar=k0,
                                                  AWGN_var=1e-14, channel_noise_var=1e-14, cascaded_channels=True,
                                                  channel_est_error=True, seed=seed, v_max=10.0, energy_budget=3e5,
                                                  episode_length=200, delta_t=1.0, layout_seed=seed)
            env = envs[seed]
            invariant('nominal PDA of the environment == (0.2, 1.5, 0)', tuple(env.nominal_pda) == (b0, k0, t0))
            hw = env.sample_hardware(np.random.default_rng(10_000 + seed), s)
            b, k, t = (np.broadcast_to(np.asarray(hw[key], float), (64,)) for key in ('beta_min', 'kappa_bar', 'theta_bar'))
            pooled[s]['th'].append(t - t0); pooled[s]['lk'].append(np.log(k / k0)); pooled[s]['b'].append(b)
            if seed == 0:                                           # curves of the seed-0 hardware instance
                for l in range(64):
                    beta_l = _amp_theta((np.full(theta.shape, b[l]), np.full(theta.shape, k[l]), np.full(theta.shape, t[l])), theta)
                    ax.plot(theta, beta_l, color=BLUE, lw=0.45, alpha=0.30)
                    invariant(f'PDA curve == Eq.(2), s={s}, element {l}', np.max(np.abs(beta_l - closed(theta, b[l], k[l], t[l]))) < 1e-12)
                    # analytic extrema: beta(theta_bar + pi/2) = 1 and beta(theta_bar - pi/2) = beta_min
                    top = float(closed(t[l] + np.pi / 2, b[l], k[l], t[l])); bot = float(closed(t[l] - np.pi / 2, b[l], k[l], t[l]))
                    invariant(f'PDA extrema, s={s}, element {l}', abs(top - 1) < 1e-12 and abs(bot - b[l]) < 1e-12)
                    rows.append([s, l, b[l], k[l], t[l]])
        nom = _amp_theta((np.full(theta.shape, b0), np.full(theta.shape, k0), np.full(theta.shape, t0)), theta)
        ax.plot(theta, nom, color=INK, lw=1.3, label='nominal')
        ax.plot([], [], color=BLUE, lw=0.8, alpha=0.6, label='64 true elements (seed 0)')
        ax.set_title(f'$s={s:g}$', fontsize=7.5)
        ax.set_xlabel(r'phase shift $\theta$ (rad)'); ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]); ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
    axes[0].set_ylabel(r'amplitude $\beta(\theta)$')
    h_, l_ = axes[0].get_legend_handles_labels()          # legend below the panels: it never covers a curve
    fig.legend(h_, l_, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07))
    # generator moments (pooled over the 5 seeds = 320 elements per level): sample std within +-15% of its target
    for s in spreads:
        th = np.concatenate(pooled[s]['th']); lk = np.concatenate(pooled[s]['lk']); bb = np.concatenate(pooled[s]['b'])
        invariant(f'hardware generator: sd(theta_bar) ~ 0.5 s, s={s}', 0.85 <= th.std(ddof=1) / (0.5 * s) <= 1.15, f'ratio={th.std(ddof=1) / (0.5 * s):.3f}')
        invariant(f'hardware generator: sd(log kappa) ~ 0.3 s, s={s}', 0.85 <= lk.std(ddof=1) / (0.3 * s) <= 1.15, f'ratio={lk.std(ddof=1) / (0.3 * s):.3f}')
        invariant(f'hardware generator: beta_min in [0.05,0.95], s={s}', bb.min() >= 0.05 - 1e-12 and bb.max() <= 0.95 + 1e-12)
    save(fig, out, 'fig_pda_model')
    write_csv(out / 'csv' / 'fig_pda_model.csv', ['spread', 'element', 'beta_min', 'kappa', 'theta_bar'], rows)


# ------------------------------------------------------------------------------------------ Fig: headroom
HEAD_ROWS = [('ao_ideal', 'AO, ideal-RIS model', 'model'), ('ao_calibrated', 'AO, true PDA curve', 'model'),
             ('ao_nominal_perfcsi', 'AO, perfect CSI', 'diag'), ('ao_calibrated_perfcsi', 'AO, true PDA, perfect CSI', 'diag'),
             ('ao_decode_nominal', 'Decode-aware AO', 'decode'), ('ao_decode_calibrated', 'Decode-aware AO, true PDA', 'decode'),
             ('cem6', 'AO+CEM, 6 probes', 'probe'), ('cem15', 'AO+CEM, 15 probes', 'probe'),
             ('cem45', 'AO+CEM, 45 probes', 'probe'), ('best_of_45', 'Random search, 45 probes', 'probe')]


def forest(ax, rows, get, ref_label=None, xlabel=''):
    """rows: [(key, label, group)]; get(key) -> per-seed paired differences (array)."""
    y = np.arange(len(rows))[::-1]
    for yi, (key, lab, grp) in zip(y, rows):
        col, mk, _ = GROUP[grp]
        d = get(key)
        m, h = tci(d)
        ax.scatter(d, np.full(len(d), yi), s=7, color=col, alpha=0.45, linewidths=0, zorder=2)        # per-seed points
        ax.errorbar([m], [yi], xerr=[h], fmt=mk, ms=4.2, color=col, mec=INK, mew=0.4, ecolor=col, elinewidth=1.1,
                    capsize=2, zorder=3)
    ax.axvline(0, color=INK, lw=0.7, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows])
    ax.set_xlabel(xlabel)
    ax.grid(axis='y', alpha=0)


def fig_headroom(data, out):
    d = load(data / '03_headroom' / 'headroom_v5_FINAL_code_d66def0.json')
    if d is None:
        return print('  skip fig_headroom (data missing)')
    seeds = 5
    spreads = ['0.0', '1.0', '2.0']
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0), sharey=True, sharex=True)
    rows_csv = []
    for ax, sp in zip(axes, spreads):
        ps = [d['per_seed'][f'{sp}|{s}'] for s in range(seeds)]
        for key, lab, grp in HEAD_ROWS:
            x = paired(ps, key, 'ao_nominal')
            m, h = tci(x)
            st = d['summary'][sp][key]['paired_decode_pct']
            check(f'headroom paired mean, s={sp}, {key}', m, st['mean'])
            check(f'headroom paired CI, s={sp}, {key}', h, st['ci95'])
            rows_csv.append([sp, key, m, h] + list(x))
        forest(ax, HEAD_ROWS, lambda k, ps=ps: paired(ps, k, 'ao_nominal'), xlabel='')
        ax.set_title(f'$s={float(sp):g}$', fontsize=7.5)
    handles = [plt.Line2D([], [], marker=GROUP[g][1], color=GROUP[g][0], mec=INK, mew=0.4, ls='', ms=4.2, label=GROUP[g][2]) for g in GROUP]
    # one shared x-label (three per-panel labels collided), legend underneath it, all inside reserved margins
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.supxlabel('decode-rate gain vs nominal AO (pp)', y=0.085, fontsize=7.5)
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, 'fig_headroom')
    write_csv(out / 'csv' / 'fig_headroom.csv', ['spread', 'method', 'mean_pp', 'ci95_pp'] + [f'seed{s}' for s in range(seeds)], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: calibration
def fig_calibration(curves, data, out):
    d = load(curves / 'calibration.json')
    if d is None:
        return print('  skip fig_calibration (calibration.json missing)')
    errs = d['config']['errors']
    cells = [k for k in d['cells']]
    fig, axes = plt.subplots(1, len(cells), figsize=(3.5 if len(cells) == 1 else 7.16, 2.4), sharey=True, squeeze=False)
    rows_csv = []
    v5 = load(data / '03_headroom' / 'headroom_v5_FINAL_code_d66def0.json')
    for ax, key in zip(axes[0], cells):
        c = d['cells'][key]
        ps = c['per_seed']
        sp = c['spread']
        for fam, lab, col, mk, ls in (('maxmin', 'max-min AO', BLUE, 'o', '-'), ('decode', 'decode-aware AO', AQUA, 's', '--')):
            mu, hw = [], []
            for e in errs:
                x = paired(ps, f'{fam}_e{e:g}', 'ao_nominal')
                m, h = tci(x)
                st = c['summary'][f'{fam}_e{e:g}']['paired_decode_pct']
                check(f'calibration {fam} e={e:g} mean, {key}', m, st['mean'])
                check(f'calibration {fam} e={e:g} CI, {key}', h, st['ci95'])
                mu.append(m); hw.append(h); rows_csv.append([key, fam, e, m, h] + list(x))
            ax.errorbar(errs, mu, yerr=hw, fmt=mk + ls, ms=3.8, color=col, mec=INK, mew=0.4, elinewidth=0.9, capsize=2, lw=1.1, label=lab)
        xn = paired(ps, 'decode_nominal', 'ao_nominal'); mn, hn = tci(xn)
        ax.axhspan(mn - hn, mn + hn, color=AQUA, alpha=0.15, lw=0)
        ax.axhline(mn, color=AQUA, lw=0.7, ls=':')
        ax.text(errs[-1], mn, ' nominal-model\n decode-aware', va='bottom', ha='right', fontsize=5.8, color=MUTED)
        ax.axhline(0, color=INK, lw=0.7)
        ax.set_xlabel(r'calibration error $e$ (units of the $s=1$ hardware spread)')
        ax.set_title(f'$s={sp:g}$', fontsize=7.5)
        # cross-stage consistency (soft): e=0 uses the true curve, so it should agree with the headroom stage's
        # true-PDA rows within their joint 95% interval (different slot streams, same estimand)
        k5 = f'{sp:.1f}'
        if v5 is not None and k5 in v5['summary']:
            for fam, k_v5 in (('maxmin', 'ao_calibrated'), ('decode', 'ao_decode_calibrated')):
                m0 = tci(paired(ps, f'{fam}_e0', 'ao_nominal'))
                m1 = v5['summary'][k5][k_v5]['paired_decode_pct']
                invariant(f'cross-stage consistency e=0 vs headroom_v5 {k_v5}, s={sp:g}',
                          abs(m0[0] - m1['mean']) <= m0[1] + m1['ci95'], f'{m0[0]:.2f} vs {m1["mean"]:.2f}', hard=False)
    axes[0][0].set_ylabel('decode-rate gain vs nominal AO (pp)')
    axes[0][0].legend(frameon=False, loc='upper right')
    fig.tight_layout()
    save(fig, out, 'fig_calibration')
    write_csv(out / 'csv' / 'fig_calibration.csv', ['cell', 'family', 'error', 'mean_pp', 'ci95_pp'] + [f'seed{s}' for s in range(len(ps))], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: probes
def fig_probes(curves, out):
    d = load(curves / 'probes.json')
    if d is None:
        return print('  skip fig_probes (probes.json missing)')
    budgets = d['config']['budgets']
    key = list(d['cells'])[0]
    c = d['cells'][key]
    ps = c['per_seed']
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    rows_csv, curve = [], {}
    for fam, lab, col, mk, ls in (('cem_nominal', 'AO+CEM from max-min AO', ORANGE, 'D', '-'), ('cem_decode', 'AO+CEM from decode-aware AO', BLUE, 'o', '--')):
        mu, hw = [], []
        for B in budgets:
            x = paired(ps, f'{fam}_B{B}', 'ao_nominal')
            m, h = tci(x)
            st = c['summary'][f'{fam}_B{B}']['paired_decode_pct']
            check(f'probes {fam} B={B} mean', m, st['mean']); check(f'probes {fam} B={B} CI', h, st['ci95'])
            mu.append(m); hw.append(h); rows_csv.append([fam, B, m, h] + list(x))
        curve[fam] = np.array(mu)
        ax.errorbar(budgets, mu, yerr=hw, fmt=mk + ls, ms=3.8, color=col, mec=INK, mew=0.4, elinewidth=0.9, capsize=2, lw=1.1, label=lab)
    xd = paired(ps, 'decode_nominal', 'ao_nominal'); md, hd = tci(xd)
    st = c['summary']['decode_nominal']['paired_decode_pct']
    check('probes decode_nominal mean', md, st['mean']); check('probes decode_nominal CI', hd, st['ci95'])
    ax.axhspan(md - hd, md + hd, color=AQUA, alpha=0.18, lw=0)
    ax.axhline(md, color=AQUA, lw=1.1, label='decode-aware AO (no probes)')
    ax.axhline(0, color=INK, lw=0.7)
    # invariants: with budget 1 the best measured candidate IS the start, so results must coincide exactly
    for fam, start in (('cem_nominal', 'ao_nominal'), ('cem_decode', 'decode_nominal')):
        if 1 in budgets:
            diff = max(abs(s[f'{fam}_B1']['decode_pct'] - s[start]['decode_pct']) for s in ps)
            invariant(f'{fam} with B=1 equals its starting solution', diff < 1e-9, f'max |diff| = {diff:.3g} pp')
    # break-even: first budget at which the CEM-from-max-min curve reaches the zero-probe decode-aware mean (log2 interpolation)
    be, y = None, curve['cem_nominal']
    for i in range(1, len(budgets)):
        if y[i - 1] < md <= y[i]:
            f = (md - y[i - 1]) / (y[i] - y[i - 1])
            be = float(2 ** (np.log2(budgets[i - 1]) + f * (np.log2(budgets[i]) - np.log2(budgets[i - 1]))))
            break
    CHECKS.append(dict(name='probes break-even budget (measurements)', value=be, ok=True, hard=False))
    if be is not None:
        ax.axvline(be, color=GREY, lw=0.7, ls=':')
        ax.annotate(f' $B^\\star\\!\\approx\\!{be:.0f}$', xy=(be, 0.12), xycoords=('data', 'axes fraction'), fontsize=6.5, va='bottom', ha='left', color=MUTED)
    ax.set_xscale('log', base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('true-rate measurements per slot $B$'); ax.set_ylabel('decode-rate gain vs nominal AO (pp)')
    ax.legend(frameon=False, loc='upper left', fontsize=6.5)
    fig.tight_layout()
    save(fig, out, 'fig_probes')
    write_csv(out / 'csv' / 'fig_probes.csv', ['family', 'B', 'mean_pp', 'ci95_pp'] + [f'seed{s}' for s in range(len(ps))], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: regime map
def fig_regime(curves, out):
    d = load(curves / 'regime.json')
    if d is None:
        return print('  skip fig_regime (regime.json missing)')
    powers, rmins = d['config']['powers'], d['config']['rmins']
    B = 46
    gain = np.full((len(rmins), len(powers)), np.nan); diff = gain.copy(); sig = np.zeros(gain.shape, int)
    rows_csv = []
    for i, rm in enumerate(rmins):
        for j, pw in enumerate(powers):
            key = f'power={pw:g}|r_min={rm:g}'
            if key not in d['cells']:
                continue
            c = d['cells'][key]
            ps = c['per_seed']
            g = paired(ps, 'decode_nominal', 'ao_nominal'); gm, gh = tci(g)
            check(f'regime gain mean {key}', gm, c['summary']['decode_nominal']['paired_decode_pct']['mean'])
            df = paired(ps, 'decode_nominal', f'cem_nominal_B{B}'); dm, dh = tci(df)
            gain[i, j], diff[i, j] = gm, dm
            sig[i, j] = 0 if abs(dm) <= dh else (1 if dm > 0 else -1)
            rows_csv.append([pw, rm, gm, gh, dm, dh, int(sig[i, j])])
    seqc = LinearSegmentedColormap.from_list('seq', ['#f3f7fc', BLUE])
    divc = LinearSegmentedColormap.from_list('div', [ORANGE, '#ececea', BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), constrained_layout=True)
    for ax, M, cmap, ttl, norm in (
            (axes[0], gain, seqc, '(a) decode-aware AO gain over nominal AO (pp)', None),
            (axes[1], diff, divc, f'(b) decode-aware AO minus AO+CEM, $B={B}$ (pp)',
             TwoSlopeNorm(vcenter=0, vmin=min(np.nanmin(diff), -1), vmax=max(np.nanmax(diff), 1)))):
        im = ax.imshow(M, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(powers))); ax.set_xticklabels([f'{p:g}' for p in powers])
        ax.set_yticks(range(len(rmins))); ax.set_yticklabels([f'{r:g}' for r in rmins])
        ax.set_xlabel('device transmit power (dBm)'); ax.set_ylabel(r'$R_{\min}$ (bit/s/Hz)'); ax.set_title(ttl, fontsize=7.2)
        ax.grid(False)
        for i in range(len(rmins)):
            for j in range(len(powers)):
                if np.isnan(M[i, j]):
                    continue
                mark = '' if M is gain else {1: '*', -1: '*', 0: ''}[sig[i, j]]
                ax.text(j, i, f'{M[i, j]:+.1f}{mark}', ha='center', va='center', fontsize=6.2, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.text(0.5, -0.045, '* 95% $t$-interval of the paired difference excludes 0 (5 seeds).\n'
             'Blue: decode-aware AO better; orange: probing better.', ha='center', va='top', fontsize=6, color=MUTED)
    save(fig, out, 'fig_regime')
    write_csv(out / 'csv' / 'fig_regime.csv', ['power_dbm', 'r_min', 'gain_pp', 'gain_ci95', 'decode_minus_cem_pp', 'diff_ci95', 'sign_certified'], rows_csv)


def _same_as_stored(new_ps, old_ps, methods, name):
    """The shared methods use identical generators, so they must reproduce the earlier study to the last bit."""
    dmax = max(abs(a[m]['decode_pct'] - b[m]['decode_pct']) for a, b in zip(new_ps, old_ps) for m in methods)
    invariant(f'{name}: shared methods reproduce the earlier study exactly', dmax < 1e-9, f'max |diff| = {dmax:.3g} pp')


def fig_probes_tuned(curves, out, ax=None, title=None):
    d, d0 = load(curves / 'probes_tuned.json'), load(curves / 'probes.json')
    if d is None or d0 is None:
        return print('  skip fig_probes_tuned (probes_tuned.json missing)')
    budgets = d['config']['budgets']; key = list(d['cells'])[0]; c = d['cells'][key]; ps = c['per_seed']
    _same_as_stored(ps, d0['cells'][key]['per_seed'], ['ao_nominal', 'decode_nominal'] + [f'cem_nominal_B{B}' for B in budgets], 'probes_tuned')
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(3.5, 3.3), constrained_layout=True)
    rows_csv, curve = [], {}
    for fam, lab, col, mk, ls in (('cem_nominal', 'AO+CEM from max-min AO', ORANGE, 'D', '-'), ('cem_tuned', 'AO+CEM from decode-aware AO', BLUE, 'o', '--')):
        mu, hw = [], []
        for B in budgets:
            x = paired(ps, f'{fam}_B{B}', 'ao_nominal'); m, h = tci(x); st = c['summary'][f'{fam}_B{B}']['paired_decode_pct']
            check(f'probes_tuned {fam} B={B} mean', m, st['mean']); check(f'probes_tuned {fam} B={B} CI', h, st['ci95'])
            mu.append(m); hw.append(h); rows_csv.append([fam, B, m, h] + list(x))
        curve[fam] = np.array(mu)
        ax.errorbar(budgets, mu, yerr=hw, fmt=mk + ls, ms=3.8, color=col, mec=INK, mew=0.4, elinewidth=0.9, capsize=2, lw=1.1, label=lab)
    for name, lab, col, ls in (('decode_tuned', 'decode-aware AO, tuned width (no probes)', AQUA, '-'), ('decode_nominal', 'decode-aware AO, width 0.2 (no probes)', AQUA, ':')):
        x = paired(ps, name, 'ao_nominal'); m, h = tci(x); st = c['summary'][name]['paired_decode_pct']
        check(f'probes_tuned {name} mean', m, st['mean']); check(f'probes_tuned {name} CI', h, st['ci95'])
        if name == 'decode_tuned':
            ax.axhspan(m - h, m + h, color=AQUA, alpha=0.16, lw=0); md = m
        ax.axhline(m, color=col, lw=1.1, ls=ls, label=lab); rows_csv.append([name, 0, m, h] + list(x))
    for fam, start in (('cem_nominal', 'ao_nominal'), ('cem_tuned', 'decode_tuned')):
        if 1 in budgets:
            diff = max(abs(s[f'{fam}_B1']['decode_pct'] - s[start]['decode_pct']) for s in ps)
            invariant(f'{fam} with B=1 equals its starting solution (tuned study)', diff < 1e-9, f'max |diff| = {diff:.3g} pp')
    y, be = curve['cem_nominal'], None
    for i in range(1, len(budgets)):
        if y[i - 1] < md <= y[i]:
            f = (md - y[i - 1]) / (y[i] - y[i - 1]); be = float(2 ** (np.log2(budgets[i - 1]) + f * (np.log2(budgets[i]) - np.log2(budgets[i - 1])))); break
    CHECKS.append(dict(name='probes_tuned break-even budget vs tuned decode-aware AO (None = not reached within the grid)', value=be, ok=True, hard=False))
    if be is not None:
        ax.axvline(be, color=GREY, lw=0.7, ls=':'); ax.annotate(f' $B^\\star\\!\\approx\\!{be:.0f}$', xy=(be, 0.12), xycoords=('data', 'axes fraction'), fontsize=6.5, va='bottom', ha='left', color=MUTED)
    ax.set_xscale('log', base=2); ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel('true-rate measurements per slot $B$'); ax.set_ylabel('decode-rate gain vs nominal AO (pp)')
    h_, l_ = ax.get_legend_handles_labels()
    if title:
        ax.set_title(title, fontsize=7.2)
    if standalone:
        fig.legend(h_, l_, loc='outside lower center', ncol=1, frameon=False, fontsize=6)
        save(fig, out, 'fig_probes_tuned')
    write_csv(out / 'csv' / 'fig_probes_tuned.csv', ['series', 'B', 'mean_pp', 'ci95_pp'] + [f'seed{s}' for s in range(len(ps))], rows_csv)
    return h_, l_


def fig_regime_tuned(curves, out, fig=None, axes=None, titles=None):
    d, d0 = load(curves / 'regime_tuned.json'), load(curves / 'regime.json')
    if d is None or d0 is None:
        return print('  skip fig_regime_tuned (regime_tuned.json missing)')
    powers, rmins, B = d['config']['powers'], d['config']['rmins'], 46
    gain = np.full((len(rmins), len(powers)), np.nan); diff = gain.copy(); sig = np.zeros(gain.shape, int); rows_csv = []
    for i, rm in enumerate(rmins):
        for j, pw in enumerate(powers):
            key = f'power={pw:g}|r_min={rm:g}'
            if key not in d['cells']:
                continue
            c = d['cells'][key]; ps = c['per_seed']
            _same_as_stored(ps, d0['cells'][key]['per_seed'], ['ao_nominal', 'decode_nominal', f'cem_nominal_B{B}'], f'regime_tuned {key}')
            g = paired(ps, 'decode_tuned', 'ao_nominal'); gm, gh = tci(g)
            check(f'regime_tuned gain mean {key}', gm, c['summary']['decode_tuned']['paired_decode_pct']['mean'])
            df = paired(ps, 'decode_tuned', f'cem_nominal_B{B}'); dm, dh = tci(df)
            gain[i, j], diff[i, j] = gm, dm; sig[i, j] = 0 if abs(dm) <= dh else (1 if dm > 0 else -1)
            rows_csv.append([pw, rm, gm, gh, dm, dh, int(sig[i, j])])
    seqc = LinearSegmentedColormap.from_list('seq', ['#f3f7fc', BLUE]); divc = LinearSegmentedColormap.from_list('div', [ORANGE, '#ececea', BLUE])
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), constrained_layout=True)
    t1, t2 = titles or ('(a) tuned decode-aware AO gain over nominal AO (pp)', f'(b) tuned decode-aware AO minus AO+CEM, $B={B}$ (pp)')
    for ax, M, cmap, ttl, norm in ((axes[0], gain, seqc, t1, None),
                                   (axes[1], diff, divc, t2,
                                    TwoSlopeNorm(vcenter=0, vmin=min(np.nanmin(diff), -1), vmax=max(np.nanmax(diff), 1)))):
        im = ax.imshow(M, origin='lower', cmap=cmap, norm=norm, aspect='auto')
        ax.set_xticks(range(len(powers))); ax.set_xticklabels([f'{p:g}' for p in powers]); ax.set_yticks(range(len(rmins))); ax.set_yticklabels([f'{r:g}' for r in rmins])
        ax.set_xlabel('device transmit power (dBm)'); ax.set_ylabel(r'$R_{\min}$ (bit/s/Hz)'); ax.set_title(ttl, fontsize=7.2); ax.grid(False)
        for i in range(len(rmins)):
            for j in range(len(powers)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f'{M[i, j]:+.1f}' + ('' if M is gain else {1: '*', -1: '*', 0: ''}[sig[i, j]]), ha='center', va='center', fontsize=6.2, color=INK)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if standalone:
        fig.text(0.5, -0.045, '* 95% $t$-interval of the paired difference excludes 0 (5 seeds).\nBlue: decode-aware AO better; orange: probing better.', ha='center', va='top', fontsize=6, color=MUTED)
        save(fig, out, 'fig_regime_tuned')
    write_csv(out / 'csv' / 'fig_regime_tuned.csv', ['power_dbm', 'r_min', 'gain_pp', 'gain_ci95', 'decode_minus_cem_pp', 'diff_ci95', 'sign_certified'], rows_csv)


def fig_probing(curves, out):
    """One wide figure for the probing question: (a) budget curve, (b) gain map, (c) zero-probe minus 46-probe CEM map."""
    if load(curves / 'probes_tuned.json') is None or load(curves / 'regime_tuned.json') is None:
        return print('  skip fig_probing (probes_tuned.json / regime_tuned.json missing)')
    fig = plt.figure(figsize=(7.16, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.12, 1.12])
    ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
    h_, l_ = fig_probes_tuned(curves, out, ax=ax0, title='(a) probing budget, $s=1$')
    fig_regime_tuned(curves, out, fig=fig, axes=(ax1, ax2), titles=('(b) gain over nominal AO (pp)', '(c) zero-probe minus 46-probe CEM (pp)'))
    fig.legend(h_, l_, loc='outside lower center', ncol=2, frameon=False, fontsize=6.2)
    save(fig, out, 'fig_probing')


# ------------------------------------------------------------------------------------------ Fig: episode level
def fig_episode(data, out):
    files = [('aoi_v5.json', '$s=1$'), ('aoi_v5_spread2.json', '$s=2$')]
    ds = [(load(data / '05_macro_and_aoi' / f), t) for f, t in files]
    if any(d is None for d, _ in ds):
        return print('  skip fig_episode (aoi_v5 data missing)')
    rows = [(sc, ll) for sc in ('greedy_aoi_travel', 'greedy_aoi') for ll in ('ao_decode_nominal', 'ao_decode_calibrated', 'ao_calibrated')]
    lab = {'greedy_aoi_travel': 'greedy+travel', 'greedy_aoi': 'greedy', 'ao_decode_nominal': 'decode-aware AO',
           'ao_decode_calibrated': 'decode-aware AO, true PDA', 'ao_calibrated': 'AO, true PDA'}
    col = {'ao_decode_nominal': (AQUA, 's'), 'ao_decode_calibrated': (AQUA, 'v'), 'ao_calibrated': (BLUE, 'o')}
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 3.6), sharey='row')
    rows_csv = []
    for c_idx, (d, ttl) in enumerate(ds):
        seeds = d['config']['seeds']
        for r_idx, (metric, xl) in enumerate((('avg_aoi', r'$\Delta$ average AoI vs nominal AO (slots)'), ('decode_pct', r'$\Delta$ decode rate vs nominal AO (pp)'))):
            ax = axes[r_idx, c_idx]
            y = np.arange(len(rows))[::-1]
            for yi, (sc, ll) in zip(y, rows):
                x = np.array([d['per_seed'][f'{sc}+{ll}|{s}'][metric] - d['per_seed'][f'{sc}+ao_nominal|{s}'][metric] for s in range(seeds)])
                m, h = tci(x)
                # raw mean of the method must match the stored summary
                raw = np.mean([d['per_seed'][f'{sc}+{ll}|{s}'][metric] for s in range(seeds)])
                check(f'episode raw mean {metric}, {ttl}, {sc}+{ll}', raw, d['summary'][f'{sc}+{ll}'][metric]['mean'])
                cc, mk = col[ll]
                ax.scatter(x, np.full(len(x), yi), s=7, color=cc, alpha=0.45, linewidths=0, zorder=2)
                ax.errorbar([m], [yi], xerr=[h], fmt=mk, ms=4.2, color=cc, mec=INK, mew=0.4, ecolor=cc, elinewidth=1.1, capsize=2, zorder=3)
                rows_csv.append([ttl, metric, sc, ll, m, h] + list(x))
            ax.axvline(0, color=INK, lw=0.7)
            ax.set_yticks(y); ax.set_yticklabels([f'{lab[sc]}: {lab[ll]}' for sc, ll in rows], fontsize=6)
            ax.grid(axis='y', alpha=0); ax.set_xlabel(xl)
            if r_idx == 0:
                ax.set_title(ttl, fontsize=7.5)
    fig.tight_layout()
    save(fig, out, 'fig_episode')
    write_csv(out / 'csv' / 'fig_episode.csv', ['panel', 'metric', 'scheduler', 'solver', 'mean', 'ci95'] + [f'seed{s}' for s in range(5)], rows_csv)


# ------------------------------------------------------------------------------------------ Fig: residual
def fig_residual(data, out):
    h2 = load(data / '04_residual_learning' / 'residual_v2_history.json')
    h3 = load(data / '04_residual_learning' / 'residual_v3_history.json')
    if h2 is None or h3 is None:
        return print('  skip fig_residual (history missing)')
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    rows_csv = []
    for h, lab, col, mk in ((h2, 'v2: unpaired reward', ORANGE, 'D'), (h3, 'v3: paired reward, anchors, fallback', BLUE, 'o')):
        x = [e['slots'] for e in h]; g = [e['policy_reward'] - e['prior_reward'] for e in h]
        for e in h:
            if 'gain' in e:
                check(f'residual stored gain == policy - prior ({lab}, slots={e["slots"]})', e['policy_reward'] - e['prior_reward'], e['gain'], atol=1e-9)
        ax.plot(x, g, mk + '-', ms=3.5, lw=1.0, color=col, mec=INK, mew=0.4, label=lab)
        rows_csv += [[lab, a, b] for a, b in zip(x, g)]
    b3 = [e.get('best_gain', 0.0) for e in h3]
    ax.step([e['slots'] for e in h3], b3, where='post', color=BLUE, lw=3.0, alpha=0.35, label='v3: best validation snapshot', zorder=1)
    invariant('v3 best-snapshot gain is never negative (zero-residual fallback)', min(b3) >= 0.0, f'min={min(b3)}')
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xlabel('training slots'); ax.set_ylabel('reward gain over AO prior')
    ax.legend(frameon=False, loc='lower left', fontsize=5.8)
    fig.tight_layout()
    save(fig, out, 'fig_residual')
    write_csv(out / 'csv' / 'fig_residual.csv', ['variant', 'slots', 'gain'], rows_csv)


# ------------------------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(ROOT / 'data'))
    p.add_argument('--curves', default=str(ROOT / 'data' / '08_curves'))
    p.add_argument('--out', default=str(ROOT / 'figures'))
    a = p.parse_args()
    data, curves, out = Path(a.data), Path(a.curves), Path(a.out)

    # global sanity: the exact t quantile used here vs the 3-digit table used by the experiment scripts
    invariant('scipy t(0.975, df=4) == 2.776 (table used by evaluate.ci) to 3 digits', abs(stats.t.ppf(0.975, 4) - 2.776) < 5e-4)

    for fn, args in ((fig_pda_model, (out,)), (fig_headroom, (data, out)), (fig_calibration, (curves, data, out)),
                     (fig_probes, (curves, out)), (fig_regime, (curves, out)), (fig_episode, (data, out)),
                     (fig_residual, (data, out)), (fig_power, (curves, out)), (fig_robust, (curves, out)), (fig_width, (curves, out)),
                     (fig_probing, (curves, out))):
        try:
            fn(*args)
        except Exception as e:                      # keep going; the failure is recorded and fails the run at the end
            import traceback
            traceback.print_exc()
            invariant(f'{fn.__name__} ran without error', False, repr(e))
    out.mkdir(parents=True, exist_ok=True)
    hard_fail = [c for c in CHECKS if not c['ok'] and c.get('hard', True)]
    soft_fail = [c for c in CHECKS if not c['ok'] and not c.get('hard', True)]
    (out / 'figure_verification.json').write_text(json.dumps(dict(
        n_checks=len(CHECKS), n_hard_failures=len(hard_fail), n_soft_failures=len(soft_fail),
        hard_failures=hard_fail, soft_failures=soft_fail, checks=CHECKS), indent=1))
    print(f'\n{len(CHECKS)} checks: {len(hard_fail)} hard failures, {len(soft_fail)} soft warnings -> {out / "figure_verification.json"}')
    for c in hard_fail[:20]:
        print('  FAIL', c)
    sys.exit(1 if hard_fail else 0)


if __name__ == '__main__':
    main()

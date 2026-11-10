#!/usr/bin/env python3
"""Generate LaTeX tables for the paper from the curated JSON in paper/data (pure Python, no torch).

    python paper/make_tables.py            # writes paper/tables/*.tex

Every number in the manuscript tables comes from here; edit nothing by hand in paper/tables/.
Needs \\usepackage{booktabs} in main.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / 'data', ROOT / 'tables'
OUT.mkdir(exist_ok=True)

NAMES = {
    'ao_ideal': 'AO, ideal-RIS assumed', 'ao_nominal': 'AO, nominal PDA (reference)',
    'ao_cal0.25': 'AO, calibration error 0.25', 'ao_cal0.5': 'AO, calibration error 0.5',
    'ao_calibrated': 'AO, true PDA',
    'ao_nominal_perfcsi': 'AO, nominal PDA, perfect CSI', 'ao_calibrated_perfcsi': 'AO, true PDA, perfect CSI',
    'ao_decode_nominal': 'Decode-aware AO, nominal PDA', 'ao_decode_cal0.25': 'Decode-aware AO, calibration error 0.25',
    'ao_decode_calibrated': 'Decode-aware AO, true PDA',
    'best_of_45': 'Random search, 45 probes', 'cem6': 'AO+CEM, 6 probes', 'cem15': 'AO+CEM, 15 probes',
    'cem45': 'AO+CEM, 45 probes',
}
GROUPS = [
    ('Model-based, no probes', ['ao_ideal', 'ao_nominal', 'ao_cal0.25', 'ao_cal0.5', 'ao_calibrated']),
    ('Perfect channel estimate (diagnostic)', ['ao_nominal_perfcsi', 'ao_calibrated_perfcsi']),
    ('Decode-aware objective, no probes', ['ao_decode_nominal', 'ao_decode_cal0.25', 'ao_decode_calibrated']),
    ('Probing (true-rate feedback)', ['best_of_45', 'cem6', 'cem15', 'cem45']),
]


def pm(d, f='{:.1f}'):
    return (f + r'$\pm$' + f).format(d['mean'], d['ci95'])


def headroom(path, out):
    s = json.loads(Path(path).read_text())['summary']
    spreads = list(s.keys())
    rows = [r'\begin{tabular}{l' + 'rr' * len(spreads) + '}', r'\toprule',
            'Method & ' + ' & '.join(r'\multicolumn{2}{c}{spread %s}' % sp for sp in spreads) + r' \\',
            ' & ' + ' & '.join(r'Dec.\,\% & $\Delta$ vs ref.' for _ in spreads) + r' \\', r'\midrule']
    for gname, ms in GROUPS:
        rows.append(r'\multicolumn{%d}{l}{\emph{%s}} \\' % (1 + 2 * len(spreads), gname))
        for m in ms:
            cells = []
            for sp in spreads:
                v = s[sp][m]
                cells += [pm(v['decode_pct']), '--' if m == 'ao_nominal' else pm(v['paired_decode_pct'])]
            rows.append(NAMES[m] + ' & ' + ' & '.join(cells) + r' \\')
        rows.append(r'\midrule')
    rows[-1] = r'\bottomrule'
    rows.append(r'\end{tabular}')
    Path(out).write_text('\n'.join(rows) + '\n')


def residual(path_v2, path_v3, out):
    rows = [r'\begin{tabular}{lrrr}', r'\toprule', r'Variant & Final gain over prior & Best snapshot gain & Eval. slots \\', r'\midrule']
    for name, p in (('Residual SAC v2 (unpaired reward)', path_v2), ('Residual SAC v3 (paired reward, anchors, fallback)', path_v3)):
        h = json.loads(Path(p).read_text())
        last = h[-1]
        g = last.get('gain', last['policy_reward'] - last['prior_reward'])
        rows.append(f"{name} & {g:+.3f} & {('n/a' if 'best_gain' not in last else format(last['best_gain'], '+.3f'))} & {last['slots']} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    Path(out).write_text('\n'.join(rows) + '\n')


def sweeps(path, out):
    d = json.loads(Path(path).read_text())
    cols = [('ao_decode_nominal', 'Decode-aware AO'), ('ao_decode_cal0.25', 'Decode-aware, cal. 0.25'),
            ('ao_calibrated', 'AO, true PDA'), ('cem15', 'CEM 15'), ('cem45', 'CEM 45')]
    rows = [r'\begin{tabular}{l' + 'r' * (1 + len(cols)) + '}', r'\toprule',
            'Setting & AO nominal (dec.\\,\\%) & ' + ' & '.join(c[1] for c in cols) + r' \\', r'\midrule']
    for key, v in d.items():
        s = v['summary']
        cells = [pm(s[m]['paired_decode_pct']) for m, _ in cols]
        rows.append(key.replace('_', r'\_') + ' & ' + f"{s['ao_nominal']['decode_pct']['mean']:.1f}" + ' & ' + ' & '.join(cells) + r' \\')
    rows += [r'\bottomrule', r'\end{tabular}']
    Path(out).write_text('\n'.join(rows) + '\n')


def aoi(path, out):
    s = json.loads(Path(path).read_text())['summary']
    rows = [r'\begin{tabular}{llrrr}', r'\toprule', r'Scheduler & RIS solver & Avg AoI & Peak AoI & Decode \% \\', r'\midrule']
    for k, v in s.items():
        sc, ll = k.split('+', 1)
        rows.append(f"{sc.replace('_', ' ')} & {ll.replace('_', ' ')} & {pm(v['avg_aoi'])} & {pm(v['peak_aoi'], '{:.0f}')} & {pm(v['decode_pct'])} \\\\")
    rows += [r'\bottomrule', r'\end{tabular}']
    Path(out).write_text('\n'.join(rows) + '\n')


if __name__ == '__main__':
    headroom(DATA / '03_headroom' / 'headroom_v5_FINAL_code_d66def0.json', OUT / 'table_headroom.tex')
    residual(DATA / '04_residual_learning' / 'residual_v2_history.json',
             DATA / '04_residual_learning' / 'residual_v3_history.json', OUT / 'table_residual.tex')
    for name, fn, rel, dst in (('sweeps', sweeps, '07_sweeps/sweeps.json', 'table_sweeps.tex'),
                               ('aoi', aoi, '05_macro_and_aoi/aoi_v5.json', 'table_aoi.tex'),
                               ('aoi_spread2', aoi, '05_macro_and_aoi/aoi_v5_spread2.json', 'table_aoi_spread2.tex')):
        if (DATA / rel).exists():
            fn(DATA / rel, OUT / dst)
        else:
            print(f'skip {name}: {rel} not in paper/data yet')
    print('wrote', *sorted(p.name for p in OUT.glob('*.tex')))

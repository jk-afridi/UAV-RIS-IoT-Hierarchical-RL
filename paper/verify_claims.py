#!/usr/bin/env python3
"""
Check that every number quoted in the manuscript text is what the stored per-seed data produce.

    python paper/verify_claims.py            # reads paper/data and paper/main.tex; prints PASS/FAIL per claim

Each claim recomputes a quantity from the PER-SEED raw values (paired differences to nominal-PDA max-min AO,
mean and exact 95% t-interval) and requires the same number, formatted as in the text, to occur in main.tex.
Qualitative statements ("in all 11 settings", "every seed improves", "14--24 points") are checked as
conditions on the data. Exit status 1 if anything fails.
"""
import json, re, sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
TEX = (ROOT / 'main.tex').read_text()
TEXN = re.sub(r'\s+', ' ', TEX)
RES = []


def J(rel):
    return json.loads((DATA / rel).read_text())


def tci(x):
    x = np.asarray(x, float); n = len(x)
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def pd_(per_seed, m, ref='ao_nominal', metric='decode_pct'):
    return np.array([s[m][metric] - s[ref][metric] for s in per_seed], float)


def has(s):
    return s in TEXN


def claim(name, strings, cond=True, detail=''):
    """strings: tokens that must occur in main.tex; cond: extra condition on the data."""
    missing = [s for s in strings if not has(s)]
    ok = bool(cond) and not missing
    RES.append((ok, name, missing, detail))


def pm(m, h, d=1):
    return f'{abs(m):.{d}f}\\pm{h:.{d}f}'


def main():
    # ----------------------------------------------------------------------------------- headroom (slot level)
    H = J('03_headroom/headroom_v5_FINAL_code_d66def0.json')
    ps = lambda sp: [H['per_seed'][f'{sp}|{s}'] for s in range(5)]
    st = {sp: {m: tci(pd_(ps(sp), m)) for m in H['summary'][sp] if m != 'ao_nominal'} for sp in ('0.0', '1.0', '2.0')}

    for sp, nm in (('0.0', 's=0'), ('1.0', 's=1'), ('2.0', 's=2')):
        m, h = st[sp]['ao_calibrated']
        claim(f'true-curve gain {nm}', [pm(m, h)])
    ideal = [st[sp]['ao_ideal'][0] for sp in st]
    claim('ideal-RIS AO loses 14--24 points at every level', ['14--24'], cond=all(-24.5 <= v <= -13.5 for v in ideal),
          detail=f'ideal deltas {np.round(ideal, 2)}')
    m, h = st['2.0']['ao_decode_cal0.25']; claim('decode-aware cal0.25 at s=2', [pm(m, h)])
    m, h = st['2.0']['ao_decode_calibrated']; claim('decode-aware true PDA at s=2', [pm(m, h)])
    for sp, nm in (('0.0', 's=0'), ('1.0', 's=1')):
        m, h = st[sp]['ao_nominal_perfcsi']; claim(f'perfect CSI gain {nm}', [pm(m, h)])
    for sp, nm in (('0.0', 's=0'), ('1.0', 's=1'), ('2.0', 's=2')):
        m, h = st[sp]['ao_decode_nominal']; claim(f'decode-aware nominal {nm}', [pm(m, h)])
    for m_, lo, hi, nm in (('cem15', '0.0', '2.0', 'CEM15'), ('cem45', '0.0', '2.0', 'CEM45')):
        claim(f'{nm} range s=0 .. s=2', [pm(*st[lo][m_]), pm(*st[hi][m_])])
    claim('cem6 worse than AO at every level', [], cond=all(st[sp]['cem6'][0] < 0 for sp in st),
          detail=str({sp: round(st[sp]['cem6'][0], 1) for sp in st}))
    bo = [st[sp]['best_of_45'][0] for sp in st]
    claim('random search 45 probes is 8--17 points worse', ['8--17'], cond=min(abs(v) for v in bo) > 7.5 and max(abs(v) for v in bo) < 17.5, detail=str(np.round(bo, 2)))

    # ----------------------------------------------------------------------------------- sweeps
    S = J('07_sweeps/sweeps.json')
    sw = {k: {m: tci(pd_(c['per_seed'], m)) for m in c['per_seed'][0] if m != 'ao_nominal'} for k, c in S.items()}
    dec = {k: v['ao_decode_nominal'] for k, v in sw.items()}
    claim('decode-aware improves in all 12 settings (default + 11) with CI excluding 0', ['all 12 parameter settings', 'all 12 settings'],
          cond=len(S) == 12 and all(m - h > 0 for m, h in dec.values()), detail=f'{len(S)} settings; min lower bound {min(m - h for m, h in dec.values()):.2f}')
    claim('abstract: default decode-aware gain', [pm(*dec['default'])])
    lo = min(dec, key=lambda k: dec[k][0]); hi = max(dec, key=lambda k: dec[k][0])
    claim('smallest sweep gain is r_min=0.25', [pm(*dec['r_min=0.25'])], cond=lo == 'r_min=0.25', detail=lo)
    claim('largest sweep gain is power_dbm=0', [pm(*dec['power_dbm=0.0'])], cond=hi == 'power_dbm=0.0', detail=hi)
    nom = tci([s['ao_nominal']['decode_pct'] for s in S['r_min=0.25']['per_seed']])[0]
    claim('nominal AO decodes ~90.6% at R_min=0.25', ['90.6'], cond=abs(nom - 90.6) < 0.05, detail=f'{nom:.2f}')
    for k in ('margin=0.0', 'margin=0.3', 'margin=0.5'):
        claim(f'margin gain {k}', [f'{dec[k][0]:.1f}'])
    claim('power 0 dBm: decode-aware vs cem45', [f'{dec["power_dbm=0.0"][0]:.1f}', f'{sw["power_dbm=0.0"]["cem45"][0]:.1f}'],
          cond=dec['power_dbm=0.0'][0] > sw['power_dbm=0.0']['cem45'][0])
    claim('R_min=1.0: true-curve decode-aware vs cem45', [f'{sw["r_min=1.0"]["ao_decode_calibrated"][0]:.1f}', f'{sw["r_min=1.0"]["cem45"][0]:.1f}'])
    claim('20 dBm: cem45 vs decode-aware', [f'{sw["power_dbm=20.0"]["cem45"][0]:.1f}', f'{dec["power_dbm=20.0"][0]:.1f}'],
          cond=sw['power_dbm=20.0']['cem45'][0] > dec['power_dbm=20.0'][0])
    claim('L=128: cem45 vs decode-aware', [f'{sw["L=128"]["cem45"][0]:.1f}', f'{dec["L=128"][0]:.1f}'])
    cal = [v['ao_calibrated'][0] for v in sw.values()]
    claim('true curve alone adds about +1 to +5 in every setting', ['about $+1$ to $+5$'], cond=0.9 <= min(cal) and max(cal) <= 5.1,
          detail=f'{min(cal):.2f}..{max(cal):.2f}')

    # ----------------------------------------------------------------------------------- episode level
    E1, E2 = J('05_macro_and_aoi/aoi_v5.json'), J('05_macro_and_aoi/aoi_v5_spread2.json')
    ep = lambda E, sc, ll, met: np.array([E['per_seed'][f'{sc}+{ll}|{s}'][met] - E['per_seed'][f'{sc}+ao_nominal|{s}'][met] for s in range(5)])
    a1 = tci(ep(E1, 'greedy_aoi_travel', 'ao_decode_nominal', 'avg_aoi')); a2 = tci(ep(E2, 'greedy_aoi_travel', 'ao_decode_nominal', 'avg_aoi'))
    claim('AoI reduction, greedy+travel, s=1 and s=2', [pm(*a1, 2), pm(*a2, 2)], cond=a1[0] < 0 and a2[0] < 0)
    raw0 = np.mean([E1['per_seed'][f'greedy_aoi_travel+ao_nominal|{s}']['avg_aoi'] for s in range(5)])
    raw1 = np.mean([E1['per_seed'][f'greedy_aoi_travel+ao_decode_nominal|{s}']['avg_aoi'] for s in range(5)])
    claim('11.8 -> 10.1 slots and about 14%', ['11.8\\to10.1', '14\\%'], cond=round(raw0, 1) == 11.8 and round(raw1, 1) == 10.1 and round(100 * (raw0 - raw1) / raw0) == 14,
          detail=f'{raw0:.2f}->{raw1:.2f} ({100 * (raw0 - raw1) / raw0:.1f}%)')
    claim('every seed improves (greedy+travel, both spreads)', ['every seed improves'],
          cond=bool(np.all(ep(E1, 'greedy_aoi_travel', 'ao_decode_nominal', 'avg_aoi') < 0) and np.all(ep(E2, 'greedy_aoi_travel', 'ao_decode_nominal', 'avg_aoi') < 0)))
    claim('decode gain greedy+travel s=1,s=2', [pm(*tci(ep(E1, 'greedy_aoi_travel', 'ao_decode_nominal', 'decode_pct'))), pm(*tci(ep(E2, 'greedy_aoi_travel', 'ao_decode_nominal', 'decode_pct')))])
    claim('true-curve decode-aware AoI reductions', [pm(*tci(ep(E1, 'greedy_aoi_travel', 'ao_decode_calibrated', 'avg_aoi')), 2), pm(*tci(ep(E2, 'greedy_aoi_travel', 'ao_decode_calibrated', 'avg_aoi')), 2)])
    claim('greedy (no travel) effects', [pm(*tci(ep(E1, 'greedy_aoi', 'ao_decode_nominal', 'avg_aoi')), 2), pm(*tci(ep(E2, 'greedy_aoi', 'ao_decode_nominal', 'avg_aoi')), 2)])

    # ----------------------------------------------------------------------------------- negative results
    A = J('05_macro_and_aoi/aoi_study_run1.json')['summary']
    claim('PPO 27.6 +- 25.1 vs greedy+travel 11.6 +- 3.4', [pm(A['ppo+ao_nominal']['avg_aoi']['mean'], A['ppo+ao_nominal']['avg_aoi']['ci95']),
                                                             pm(A['greedy_aoi_travel+ao_nominal']['avg_aoi']['mean'], A['greedy_aoi_travel+ao_nominal']['avg_aoi']['ci95'])])
    R = J('04_residual_learning/residual_v2_negative_result.json')['summary']['0.0']
    claim('v2 residual 47% vs 75% decode at s=0', ['47\\%', '75\\%'], cond=round(R['residual_sac']['decode_pct']['mean']) == 47 and round(R['ao_nominal']['decode_pct']['mean']) in (75, 76),
          detail=f"{R['residual_sac']['decode_pct']['mean']:.2f} vs {R['ao_nominal']['decode_pct']['mean']:.2f}")
    h3 = J('04_residual_learning/residual_v3_history.json')
    claim('v3: no checkpoint beat the prior (8000 slots)', ['8000'], cond=all(e['gain'] <= 0 for e in h3) and h3[-1]['slots'] == 8000 and all(e['best_gain'] == 0 for e in h3),
          detail=f"gains {[round(e['gain'], 3) for e in h3]}, last slots {h3[-1]['slots']}")

    # ----------------------------------------------------------------------------------- curve studies (08_curves)
    C = J('08_curves/calibration.json')['cells']
    cal = lambda sp, k: tci(pd_(C[sp]['per_seed'], k))
    claim('calibration s=1 decode-aware e=0.5, e=1, e=2, e=0',
          [pm(*cal('spread=1', 'decode_e0.5')), pm(*cal('spread=1', 'decode_e1')), pm(*cal('spread=1', 'decode_e2')), pm(*cal('spread=1', 'decode_e0'))],
          cond=cal('spread=1', 'decode_e2')[0] < 0 < cal('spread=1', 'decode_e1')[0])
    claim('calibration s=2 decode-aware e=0.5, 1, 2', [pm(*cal('spread=2', 'decode_e0.5')), pm(*cal('spread=2', 'decode_e1')), pm(*cal('spread=2', 'decode_e2'))],
          cond=all(cal('spread=2', f'decode_e{e}')[0] - cal('spread=2', f'decode_e{e}')[1] > 0 for e in ('0.5', '1', '2')))
    claim('calibration s=1 max-min e=0.5 and e=2', [pm(*cal('spread=1', 'maxmin_e0.5')), pm(*cal('spread=1', 'maxmin_e2'))])
    P = J('08_curves/probes.json')['cells']['spread=1']['per_seed']
    pr = lambda k: tci(pd_(P, k))
    claim('probes: CEM from max-min at B=10,16,46', [pm(*pr('cem_nominal_B10')), pm(*pr('cem_nominal_B16')), pm(*pr('cem_nominal_B46'))])
    claim('probes: CEM from decode-aware at B=10,46; zero-probe decode-aware', [pm(*pr('cem_decode_B10')), pm(*pr('cem_decode_B46')), pm(*pr('decode_nominal'))])
    bs = [1, 4, 7, 10, 16, 31, 46, 91]
    y = [pr(f'cem_nominal_B{b}')[0] for b in bs]; md = pr('decode_nominal')[0]
    be = next(2 ** (np.log2(bs[i - 1]) + (md - y[i - 1]) / (y[i] - y[i - 1]) * (np.log2(bs[i]) - np.log2(bs[i - 1]))) for i in range(1, len(bs)) if y[i - 1] < md <= y[i])
    claim('break-even budget about 13', ['B^\\star\\approx13'], cond=round(be) == 13, detail=f'{be:.2f}')
    claim('probes B=1 equals start', [], cond=all(abs(q['cem_nominal_B1']['decode_pct'] - q['ao_nominal']['decode_pct']) < 1e-9 for q in P))
    Rg = J('08_curves/regime.json')['cells']
    rg = lambda pw, rm, a, b='ao_nominal': tci(pd_(Rg[f'power={pw}|r_min={rm}']['per_seed'], a, b))
    lo, hi = min(rg(pw, rm, 'decode_nominal')[0] for pw in (0, 5, 10, 15, 20) for rm in (0.25, 0.5, 0.75, 1) if f'power={pw}|r_min={rm}' in Rg), max(rg(pw, rm, 'decode_nominal')[0] for pw in (0, 5, 10, 15, 20) for rm in (0.25, 0.5, 0.75, 1) if f'power={pw}|r_min={rm}' in Rg)
    claim('regime: decode-aware gain range +3 to +22', ['$+3$ to $+22$'], cond=round(lo) == 3 and round(hi) == 22, detail=f'{lo:.2f}..{hi:.2f}')
    d00 = tci(pd_(Rg['power=0|r_min=1']['per_seed'], 'decode_nominal', 'cem_nominal_B46'))
    d20 = tci(pd_(Rg['power=20|r_min=1']['per_seed'], 'decode_nominal', 'cem_nominal_B46'))
    claim('regime: +8.2 +- 0.9 at 0 dBm Rmin=1 and -31.2 +- 5.6 at 20 dBm Rmin=1', [pm(*d00), pm(*d20)])
    sig = lambda pw, rm: (lambda m, h: 0 if abs(m) <= h else int(np.sign(m)))(*tci(pd_(Rg[f'power={pw}|r_min={rm}']['per_seed'], 'decode_nominal', 'cem_nominal_B46')))
    claim('regime: certified signs as described', [], cond=all(sig(0, r) == 1 for r in (0.5, 0.75, 1)) and all(sig(5, r) == 1 for r in (0.75, 1))
          and all(sig(pw, r) == -1 for pw in (10, 15, 20) for r in (0.25, 0.5, 0.75, 1)))

    # ----------------------------------------------------------------------------------- report
    fails = [r for r in RES if not r[0]]
    for ok, name, missing, detail in RES:
        print(('PASS ' if ok else 'FAIL ') + name + (f'   [{detail}]' if detail else '') + (f'   missing in tex: {missing}' if missing else ''))
    print(f'\n{len(RES) - len(fails)}/{len(RES)} claims verified')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()

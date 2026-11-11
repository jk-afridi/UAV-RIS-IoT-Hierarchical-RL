"""Unit tests for the statistics and bookkeeping behind the paper figures (no experiments are run)."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'paper'))

import make_figures as mf                                   # noqa: E402
from eval import curves                                     # noqa: E402


def test_tci_matches_textbook_formula():
    x = np.array([1.0, 2.5, 3.0, 4.5, 6.0])
    m, h = mf.tci(x)
    assert abs(m - x.mean()) < 1e-12
    assert abs(h - 2.7764451 * x.std(ddof=1) / np.sqrt(5)) < 1e-6                      # t_{0.975,4}


def test_tci_matches_experiment_scripts():
    """The CI stored by evaluate.ci (3-digit t table) and the exact one used for figures agree to 0.2%."""
    from eval.evaluate import ci
    x = np.random.default_rng(0).normal(size=5)
    (m1, h1), (m2, h2) = ci(list(x)), mf.tci(x)
    assert abs(m1 - m2) < 1e-12 and abs(h1 - h2) <= 2e-3 * h2


def test_paired_difference_is_per_seed():
    ps = [{'a': {'decode_pct': 70.0}, 'b': {'decode_pct': 75.0}}, {'a': {'decode_pct': 60.0}, 'b': {'decode_pct': 62.0}}]
    assert np.allclose(mf.paired(ps, 'b', 'a'), [5.0, 2.0])


def test_check_flags_mismatch():
    mf.CHECKS.clear()
    assert mf.check('same', 1.0, 1.0005) and not mf.check('different', 1.0, 1.1)
    assert [c['ok'] for c in mf.CHECKS] == [True, False]
    mf.CHECKS.clear()


def test_probe_budgets_are_exactly_spendable():
    for B in curves.PROBE_BUDGETS + [curves.REGIME_B]:
        assert curves.n_measurements(B) == B
    assert curves.n_measurements(1) == 1 and curves.n_measurements(6) == 4          # 6-1=5 -> 3 iterations x 1


def test_method_generators_are_independent_of_each_other():
    class A:  # minimal stand-in for the parsed args
        ao_sweeps, margin = 1, 0.15
    s1 = curves.Slot(None, None, 3, 1.0, 7, A()); s2 = curves.Slot(None, None, 3, 1.0, 7, A())
    assert s1.rng('x').integers(1 << 30) == s2.rng('x').integers(1 << 30)           # reproducible
    assert s1.rng('x').integers(1 << 30) != s1.rng('y').integers(1 << 30)           # independent across methods
    s3 = curves.Slot(None, None, 3, 1.0, 8, A())
    assert s1.rng('x').integers(1 << 30) != s3.rng('x').integers(1 << 30)           # independent across slots


def test_summarise_pairs_against_reference():
    per_seed = {'k': {s: {'ao_nominal': {'decode_pct': 70.0 + s}, 'm': {'decode_pct': 75.0 + s + 0.5 * (s % 2)}} for s in range(5)}}
    out = curves.summarise(per_seed, ['k'], 5)
    d = np.array([5.0 + 0.5 * (s % 2) for s in range(5)])
    assert abs(out['k']['m']['paired_decode_pct']['mean'] - d.mean()) < 1e-12
    assert 'paired_decode_pct' not in out['k']['ao_nominal']

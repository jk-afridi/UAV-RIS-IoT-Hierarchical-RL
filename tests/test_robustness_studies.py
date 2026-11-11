"""Checks for the quantisation / tau / CSI-error additions (no training, CPU only)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from baselines.ris_baselines import phase_grid, ao_solution, ao_decode, cascaded_csi, _GRID  # noqa: E402
from train.train_residual import build_parser, make_slot_env, sample_slot                      # noqa: E402


def _slot(seed=0):
    a = build_parser().parse_args([])
    env = make_slot_env(seed, a)
    rng = np.random.default_rng(seed)
    sample_slot(env, rng, a.max_dist, fixed_dist=5.0)
    return env, rng


def test_phase_grid_sizes_and_subset():
    for b in range(1, 6):
        g = phase_grid(b)
        assert len(g) == 2 ** b and set(np.round(g, 12)) <= set(np.round(_GRID, 12))
        assert np.allclose(np.diff(g), 2 * np.pi / 2 ** b)          # uniformly spaced over the circle
    assert phase_grid(6) is _GRID and phase_grid(None) is _GRID


def test_quantised_solution_lies_on_the_restricted_grid():
    env, rng = _slot()
    for b in (1, 2, 3):
        th, _ = ao_solution(env, np.random.default_rng(1), model='nominal', sweeps=1, bits=b)
        g = phase_grid(b)
        assert np.all(np.min(np.abs(th[:, None] - g[None, :]), axis=1) < 1e-12)


def test_six_bits_and_default_tau_are_the_unrestricted_solver():
    env, _ = _slot()
    Dh = cascaded_csi(env, np.random.default_rng(3))
    t0, _ = ao_decode(env, np.random.default_rng(4), Dh=Dh, sweeps=1)
    t1, _ = ao_decode(env, np.random.default_rng(4), Dh=Dh, sweeps=1, bits=6, tau_frac=0.2)
    assert np.array_equal(t0, t1)

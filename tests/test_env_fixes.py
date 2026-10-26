"""Regression tests for the problems found in the audit. numpy-only (needs `gym`).
Run:  python -m pytest tests -q     (from repo root)"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core.aoi_environment import AoIUAVRISEnvironment          # noqa: E402
from baselines.ris_baselines import ao_ris, random_action        # noqa: E402


def make(layout_seed=3, **kw):
    return AoIUAVRISEnvironment(num_antennas=4, num_RIS_elements=64, num_users=4, beta_min=0.2,
                                AWGN_var=1e-14, channel_noise_var=1e-14, cascaded_channels=True,
                                channel_est_error=True, seed=0, episode_length=200, r_min=0.5,
                                k_rice=10.0, k_rice_bs=1.0, iot_tx_power_dbm=10.0,
                                layout_seed=layout_seed, **kw)


def _gain_at(env, cid, offset):
    cx, cy = env.ris_positions[cid][:2]
    g = []
    for _ in range(40):
        env.high_level_reset()
        env.uav_pos = np.array([np.clip(cx + offset, 0, 100), cy, 15.0])
        env.high_level_step(np.zeros(2), cid)            # the call that used to discard the position
        g.append(np.mean(np.abs(env.H_2) ** 2))
    return np.mean(g)


def test_uav_position_changes_channel():
    env = make()
    near, far = _gain_at(env, 3, 0.0), _gain_at(env, 3, 40.0)
    assert near > 5 * far, (near, far)        # before the fix: near == far (position ignored)


def test_decode_depends_on_distance():
    env, rng = make(), np.random.default_rng(0)
    cid = 3; cx, cy = env.ris_positions[cid][:2]
    def dec(off):
        out = []
        for _ in range(15):
            env.high_level_reset(); env.uav_pos = np.array([np.clip(cx + off, 0, 100), cy, 15.0])
            env.high_level_step(np.zeros(2), cid)
            out.append(np.mean(env.compute_rates_from_action(ao_ris(env, rng)) >= env.r_min))
        return np.mean(out)
    assert dec(0.0) > dec(30.0) + 0.2


def test_layout_seed():
    a, b, c = make(1), make(2), make(1)
    assert not np.allclose(a.ris_positions, b.ris_positions)
    assert np.allclose(a.ris_positions, c.ris_positions)


def test_phase_amplitude_scaling_does_not_change_rate():
    """Documents the audit finding: the rate depends on arctan2(Re, Im) only, so scaling the
    phase action by a Beta-space amplitude cannot change the reward."""
    env = make(); env.set_current_ris_group(2); env._update_channels_for_position(); env._build_initial_state()
    a = random_action(env, np.random.default_rng(1)); r0 = env.compute_rates_from_action(a)
    b = a.copy(); b[-128:] *= 0.3
    assert np.allclose(env.compute_rates_from_action(b), r0)

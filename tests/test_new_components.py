"""Tests for the hardware-mismatch / residual / macro-PPO components (needs torch + gym).
Run on Colab:  python -m pytest tests -q"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from core.aoi_environment import AoIUAVRISEnvironment                                   # noqa: E402
from baselines.ris_baselines import (ao_ris, cascaded_csi, feature_dim, slot_features,   # noqa: E402
                                     ao_solution, residual_action, cem_ris)


def make(seed=2):
    return AoIUAVRISEnvironment(num_antennas=4, num_RIS_elements=64, num_users=4, beta_min=0.2,
                                AWGN_var=1e-14, channel_noise_var=1e-14, cascaded_channels=True,
                                channel_est_error=True, seed=0, episode_length=200, r_min=0.5,
                                k_rice=10.0, k_rice_bs=1.0, iot_tx_power_dbm=10.0, layout_seed=seed)


def _slot(env, rng, cid=1, dist=3.0):
    cx, cy = env.ris_positions[cid][:2]
    env.set_current_ris_group(cid)
    env.uav_pos = np.array([cx + dist, cy, env.ris_height])
    env._update_channels_for_position()


def test_high_state_dim_and_layout_reset():
    env = make()
    s = env.high_level_reset()
    assert s.shape == (env.high_state_dim,) and env.high_state_dim == 3 + 2 * 10 + 40
    old = env.ris_positions.copy(); env.reset_layout(99)
    assert not np.allclose(old, env.ris_positions)


def test_hardware_override_changes_rates_but_not_nominal():
    env, rng = make(), np.random.default_rng(0)
    _slot(env, rng); a = ao_ris(env, rng, noisy_csi=False, model='nominal')
    r0 = env.compute_rates_from_action(a)
    nominal = env.nominal_pda
    env.set_hardware(**env.sample_hardware(np.random.default_rng(1), 2.0))
    assert env.nominal_pda == nominal
    assert not np.allclose(env.compute_rates_from_action(a), r0)


def test_calibrated_ao_not_worse_than_nominal_under_mismatch():
    env, rng = make(), np.random.default_rng(0)
    env.set_hardware(**env.sample_hardware(np.random.default_rng(5), 2.0))
    nom, cal = [], []
    for _ in range(20):
        _slot(env, rng); Dh = cascaded_csi(env, rng)
        nom.append(env.compute_rates_from_action(ao_ris(env, rng, Dh=Dh, model='nominal')).min())
        cal.append(env.compute_rates_from_action(ao_ris(env, rng, Dh=Dh, model='true')).min())
    assert np.mean(cal) >= np.mean(nom)


def test_cem_returns_valid_action():
    env, rng = make(), np.random.default_rng(0)
    _slot(env, rng)
    a = cem_ris(env, rng, 15)
    assert a.shape == (2 * 16 + 2 * 64,) and np.all(np.isfinite(env.compute_rates_from_action(a)))


def test_residual_policy_starts_at_physics_prior():
    from algorithms.residual_sac import ResidualSAC
    env, rng = make(), np.random.default_rng(0)
    agent = ResidualSAC(feature_dim(env), env.L, device='cpu')
    _slot(env, rng); Dh = cascaded_csi(env, rng)
    th, _ = ao_solution(env, rng, Dh=Dh, model='nominal', sweeps=1)
    _, x, delta = residual_action(env, rng, agent, Dh=Dh, deterministic=True, prior_theta=th)
    assert x.shape == (feature_dim(env),) and np.max(np.abs(delta)) < 0.2


def test_residual_update_runs():
    import torch
    from algorithms.residual_sac import ResidualSAC
    agent = ResidualSAC(32, 8, device='cpu', hidden=32)
    x, a, r = torch.randn(16, 32), torch.randn(16, 8).clamp(-1, 1), torch.rand(16)
    info = agent.update(x, a, r)
    assert all(np.isfinite(v) for v in info.values())


def test_macro_ppo_roundtrip():
    from algorithms.macro_ppo import HybridPPO
    env = make(); s = env.high_level_reset()
    ppo = HybridPPO(env.high_state_dim, env.num_ris_groups, hidden=32, minibatch=8)
    for t in range(16):
        c, vel, raw, logp, val = ppo.act(s)
        assert 0 <= c < 10 and vel.shape == (2,)
        _, done, _ = env.high_level_step(vel * env.v_max, c, build_low_state=False)
        ppo.store(s, c, raw, logp, -1.0, float(t == 15), val); s = env.get_high_level_state()
    assert 'pg' in ppo.update()

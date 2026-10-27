"""Factory for the low-level (RIS + beamforming) policy used inside episode-level studies."""
from .ris_baselines import ao_ris, cem_ris, random_action, residual_action


def make_lowlevel(name, rng, ckpt=None, device='cpu', ao_sweeps=2):
    """Returns f(env) -> low-level action for the CURRENT channel in env (call after high_level_step)."""
    if name == 'ao_nominal':
        return lambda env: ao_ris(env, rng, model='nominal', sweeps=ao_sweeps)
    if name == 'ao_ideal':
        return lambda env: ao_ris(env, rng, model='ideal', sweeps=ao_sweeps)
    if name == 'ao_calibrated':
        return lambda env: ao_ris(env, rng, model='true', sweeps=ao_sweeps)
    if name == 'cem15':
        return lambda env: cem_ris(env, rng, 15, model='nominal', ao_sweeps=ao_sweeps)
    if name == 'random':
        return lambda env: random_action(env, rng)
    if name == 'residual':
        from algorithms.residual_sac import ResidualSAC          # lazy: needs torch
        agent = ResidualSAC.load(ckpt, device=device)
        return lambda env: residual_action(env, rng, agent, deterministic=True, sweeps=ao_sweeps)[0]
    raise ValueError(f'unknown low-level policy {name}')

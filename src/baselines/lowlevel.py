"""Factory for the low-level (RIS + beamforming) policy used inside episode-level studies."""
from .ris_baselines import ao_ris, ao_decode, cem_ris, pack_action, random_action, residual_action


_DECODE_MODELS = {'ao_decode': 'nominal', 'ao_decode_nominal': 'nominal', 'ao_decode_calibrated': 'true', 'ao_decode_cal0.25': 'cal:0.25'}


def make_lowlevel(name, rng, ckpt=None, device='cpu', ao_sweeps=2):
    """Returns f(env) -> low-level action for the CURRENT channel in env (call after high_level_step)."""
    if name == 'ao_nominal':
        return lambda env: ao_ris(env, rng, model='nominal', sweeps=ao_sweeps)
    if name == 'ao_ideal':
        return lambda env: ao_ris(env, rng, model='ideal', sweeps=ao_sweeps)
    if name == 'ao_calibrated':
        return lambda env: ao_ris(env, rng, model='true', sweeps=ao_sweeps)
    if name in _DECODE_MODELS:                      # decoding-aware AO, no probes ('ao_decode' = nominal, alias)
        model = _DECODE_MODELS[name]

        def f(env):
            th, G = ao_decode(env, rng, model=model, sweeps=ao_sweeps)
            return pack_action(env, G, th)
        return f
    if name == 'ao_cal0.25':
        return lambda env: ao_ris(env, rng, model='cal:0.25', sweeps=ao_sweeps)
    if name == 'cem45':
        return lambda env: cem_ris(env, rng, 45, model='nominal', ao_sweeps=ao_sweeps)
    if name == 'cem15':
        return lambda env: cem_ris(env, rng, 15, model='nominal', ao_sweeps=ao_sweeps)
    if name == 'random':
        return lambda env: random_action(env, rng)
    if name == 'residual':
        from algorithms.residual_sac import ResidualSAC          # lazy: needs torch
        agent = ResidualSAC.load(ckpt, device=device)
        return lambda env: residual_action(env, rng, agent, deterministic=True, sweeps=ao_sweeps)[0]
    raise ValueError(f'unknown low-level policy {name}')

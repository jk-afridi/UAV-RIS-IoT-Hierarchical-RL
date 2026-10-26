"""
Non-learning RIS / beamforming baselines for the UAV-RIS AoI environment.

These exist because the original paper compared its SAC micro-agent only with
*random* RIS phases. A reviewer will immediately ask how it compares with
(i) classical alternating optimisation (AO) and (ii) plain random search with the
same oracle feedback the SAC wrapper uses. All functions return a 160-dim
low-level action in the format `env.compute_rates_from_action` expects, so every
method is scored by the *same* rate function as the DRL agent.

No torch dependency.
"""
import numpy as np

_GRID = np.linspace(-np.pi, np.pi, 64, endpoint=False)


def pack_action(env, G, theta):
    """(G: MxK complex, theta: L phases) -> flat action. The env reads the phase
    as arctan2(Re, Im), so Re=sin(theta), Im=cos(theta) gives angle theta."""
    M, L = env.M, env.L
    a = np.zeros(2 * M * M + 2 * L)
    a[:M * M] = G.real.ravel()
    a[M * M:2 * M * M] = G.imag.ravel()
    a[-2 * L:-L] = np.sin(theta)
    a[-L:] = np.cos(theta)
    return a


def random_action(env, rng):
    G = rng.normal(size=(env.M, env.K)) + 1j * rng.normal(size=(env.M, env.K))
    G *= np.sqrt(10.0) / np.linalg.norm(G)
    return pack_action(env, G, rng.uniform(-np.pi, np.pi, env.L))


def sac_style_reward(env, action):
    """Reward the SAC wrapper optimises: sum-rate + 50 * #decoded."""
    r = env.compute_rates_from_action(action)
    return r.sum() + 50.0 * np.sum(r >= env.r_min)


def best_of_n(env, n, rng):
    """Random search with oracle (true-rate) selection -- the mechanism hidden in
    LowLevelSACWrapper.optimize_for_cluster (best of `inner_steps` actions)."""
    best, best_r = None, -np.inf
    for _ in range(n):
        a = random_action(env, rng)
        r = sac_style_reward(env, a)
        if r > best_r:
            best, best_r = a, r
    return best


def cascaded_csi(env, rng=None, noisy=True):
    """Cascaded channels D_k = diag(h2_k) H1, shape (K, L, M). If `noisy`, add the
    same estimation error the environment injects into the agent's state."""
    D = np.stack([np.diag(env.H_2[:, k]) @ env.H_1 for k in range(env.K)])
    if noisy and env.channel_noise_var > 0:
        rng = rng or np.random
        s = np.sqrt(env.channel_noise_var / 2)
        D = D + rng.normal(0, s, D.shape) + 1j * rng.normal(0, s, D.shape)
    return D


def ao_ris(env, rng, noisy_csi=True, pda_aware=True, sweeps=3):
    """Alternating optimisation: zero-forcing combiner <-> per-element phase
    search maximising the minimum user rate (grid over 64 phases).
    pda_aware=False assumes ideal unit-modulus reflection when optimising
    (but is still *evaluated* with the true PDA amplitudes)."""
    L = env.L
    Dh = cascaded_csi(env, rng, noisy=noisy_csi)
    noise_scale = env.awgn_var / env.iot_tx_power
    amp = (lambda th: env._compute_PDA(th)) if pda_aware else (lambda th: np.ones_like(th))
    vgrid = amp(_GRID) * np.exp(1j * _GRID)
    theta = rng.uniform(-np.pi, np.pi, L)
    for _ in range(sweeps):
        v = amp(theta) * np.exp(1j * theta)
        E = np.einsum('l,klm->km', v, Dh)
        G = np.linalg.pinv(E)
        G *= np.sqrt(10.0) / np.linalg.norm(G)
        Cf = np.einsum('klm,mj->kjl', Dh, G)           # (K,K,L)
        nrm = np.linalg.norm(G, axis=0) ** 2
        for l in range(L):
            base = Cf @ v - Cf[:, :, l] * v[l]
            cand = base[:, :, None] + Cf[:, :, l][:, :, None] * vgrid[None, None, :]
            diag = np.einsum('kkg->kg', cand)
            sig = np.abs(diag) ** 2
            inter = (np.abs(cand) ** 2).sum(axis=0) - sig
            sinr = sig / (inter + noise_scale * nrm[:, None])
            j = int(np.argmax(np.log2(1 + sinr).min(axis=0)))
            theta[l], v[l] = _GRID[j], vgrid[j]
    return pack_action(env, G, theta)


RIS_POLICIES = {
    'random1':        lambda env, rng: random_action(env, rng),
    'best_of_5':      lambda env, rng: best_of_n(env, 5, rng),
    'best_of_10':     lambda env, rng: best_of_n(env, 10, rng),
    'ao_unaware':     lambda env, rng: ao_ris(env, rng, noisy_csi=True, pda_aware=False),
    'ao_aware':       lambda env, rng: ao_ris(env, rng, noisy_csi=True, pda_aware=True),
    'ao_aware_perfect': lambda env, rng: ao_ris(env, rng, noisy_csi=False, pda_aware=True),
}

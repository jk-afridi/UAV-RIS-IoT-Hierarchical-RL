"""
Non-learning RIS / beamforming baselines for the UAV-RIS AoI environment.

Every method returns a flat low-level action in the format
`env.compute_rates_from_action` expects, so ALL methods (classical and learned) are
scored by the same rate function on the same channel realisation.

Amplitude models a solver can assume (`model=`):
  'nominal' : the designer's PDA curve  (env.nominal_pda)            <- what a real design uses
  'true'    : the actual hardware (env.beta_min/kappa_bar/theta_bar)   <- "calibrated" ceiling
  'ideal'   : unit-modulus reflection (PDA ignored)

Measurement-budget baselines (`cem_ris`, `best_of_n`) model a solver that may probe the true rate
B times per slot (the same free feedback the original SAC wrapper used implicitly).

No torch dependency.
"""
import numpy as np

_GRID = np.linspace(-np.pi, np.pi, 64, endpoint=False)


# --------------------------------------------------------------------------- helpers
def pack_action(env, G, theta):
    """(G: MxK complex, theta: L phases) -> flat action. The env reads the phase as
    arctan2(Re, Im), so Re=sin(theta), Im=cos(theta) gives angle theta."""
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
    """Reward the original SAC wrapper optimised: sum-rate + 50 * #decoded."""
    r = env.compute_rates_from_action(action)
    return r.sum() + 50.0 * np.sum(r >= env.r_min)


def best_of_n(env, n, rng):
    """Random search with oracle (true-rate) selection -- the mechanism hidden in the original
    LowLevelSACWrapper.optimize_for_cluster (best of `inner_steps` actions)."""
    best, best_r = None, -np.inf
    for _ in range(n):
        a = random_action(env, rng)
        r = sac_style_reward(env, a)
        if r > best_r:
            best, best_r = a, r
    return best


def cascaded_csi(env, rng=None, noisy=True):
    """Cascaded channels D_k = diag(h2_k) H1, shape (K, L, M). If `noisy`, add the same
    estimation error the environment injects into the agent's state."""
    D = np.stack([np.diag(env.H_2[:, k]) @ env.H_1 for k in range(env.K)])
    if noisy and env.channel_noise_var > 0:
        rng = rng if rng is not None else np.random
        s = np.sqrt(env.channel_noise_var / 2)
        D = D + rng.normal(0, s, D.shape) + 1j * rng.normal(0, s, D.shape)
    return D


def _params(env, model):
    """PDA parameters (beta_min, kappa, theta_bar), each broadcast to shape (L,)."""
    if model == 'ideal':
        p = (1.0, 1.0, 0.0)                       # beta_min = 1  ->  amplitude == 1
    elif model == 'nominal':
        p = env.nominal_pda
    elif model == 'true':
        p = (env.beta_min, env.kappa_bar, env.theta_bar)
    elif isinstance(model, str) and model.startswith('cal:'):
        return _calibrated_params(env, float(model.split(':')[1]))
    else:
        raise ValueError(model)
    return tuple(np.broadcast_to(np.asarray(x, float), (env.L,)).copy() for x in p)


def _calibrated_params(env, err):
    """Per-element PDA parameters as a one-off calibration campaign would estimate them: the true
    values plus a FIXED estimation error (same for every slot of this hardware instance). `err` is
    in units of the spread-1 hardware deviation (sd: beta_min 0.15, log-kappa 0.3, theta_bar 0.5 rad),
    so err=0 is the calibrated ideal and err=1 is as wrong as a spread-1 mismatch. This is a
    sensitivity model ('how accurate must calibration be?'), not a probe-based estimator."""
    tb, tk, tt = (np.broadcast_to(np.asarray(x, float), (env.L,)).copy()
                  for x in (env.beta_min, env.kappa_bar, env.theta_bar))
    key = (round(float(err), 6), tb.tobytes(), tk.tobytes(), tt.tobytes())
    cache = getattr(env, '_cal_cache', None)
    if cache is None:
        cache = env._cal_cache = {}
    if key not in cache:
        r = np.random.default_rng(987_654 + env.L)
        cache[key] = (np.clip(tb + err * 0.15 * r.normal(size=env.L), 0.02, 0.98),
                      np.clip(tk * np.exp(err * 0.3 * r.normal(size=env.L)), 0.2, None),
                      tt + err * 0.5 * r.normal(size=env.L))
    return tuple(x.copy() for x in cache[key])


def _amp_theta(params, theta):                    # theta: (L,)
    b, k, t = params
    return (1 - b) * ((np.sin(theta - t) + 1) / 2) ** k + b


def phase_grid(bits=None):
    """Candidate phase set: the 64-point grid (bits=None or 6: the 'continuous' reference) or a b-bit subset of it."""
    if bits is None or bits >= 6:
        return _GRID
    assert 1 <= bits <= 5, bits
    return _GRID[::len(_GRID) // 2 ** bits]


def _amp_grid(params, grid=None):                 # -> (L, G)
    grid = _GRID if grid is None else grid
    b, k, t = (x[:, None] for x in params)
    return (1 - b) * ((np.sin(grid[None, :] - t) + 1) / 2) ** k + b


def zf_beamformer(Dh, v):
    """Zero-forcing combiner for effective channel E = sum_l v_l Dh[:, l, :]  (K x M)."""
    E = np.einsum('l,klm->km', v, Dh)
    G = np.linalg.pinv(E)                          # (M, K)
    return G * np.sqrt(10.0) / np.linalg.norm(G)


# --------------------------------------------------------------------------- alternating optimisation
def ao_solution(env, rng, Dh=None, noisy_csi=True, model='nominal', sweeps=3, init_theta=None,
                objective='maxmin', margin=0.15, tau_frac=0.2, bits=None):
    """Alternating optimisation: ZF combiner <-> per-element phase search maximising the minimum
    user rate. The solver ASSUMES the amplitude `model`; evaluation always uses the true hardware.
    objective='maxmin' (default): maximise the minimum rate.
    objective='decode': maximise the smooth expected number of decoded users, sum_k sigmoid((rate_k -
    r_min*(1+margin))/(0.2 r_min)), plus a small max-min tie-break (v1 used a hard count with capped rate:
    its plateaus made the element choice arbitrary and it lost ~20 pp). No probes. Returns (theta, G).
    tau_frac: sigmoid width as a fraction of r_min (default 0.2 = the value used throughout the paper).
    bits: restrict every phase to a b-bit subset of the grid (None / 6 = unrestricted; the default path is unchanged)."""
    L = env.L
    if Dh is None:
        Dh = cascaded_csi(env, rng, noisy=noisy_csi)
    params = _params(env, model)
    grid = phase_grid(bits)
    vgrid = _amp_grid(params, grid) * np.exp(1j * grid)[None, :]     # (L, G)
    noise_scale = env.awgn_var / env.iot_tx_power
    theta = rng.uniform(-np.pi, np.pi, L) if init_theta is None else np.array(init_theta, float)
    if grid is not _GRID:                                            # snap the start onto the restricted phase set
        theta = grid[np.argmin(np.abs(np.angle(np.exp(1j * (theta[:, None] - grid[None, :])))), axis=1)]
    for _ in range(sweeps):
        v = _amp_theta(params, theta) * np.exp(1j * theta)
        G = zf_beamformer(Dh, v)
        Cf = np.einsum('klm,mj->kjl', Dh, G)                          # (K,K,L): [user, combiner, element]
        nrm = np.linalg.norm(G, axis=0) ** 2
        for l in range(L):
            base = Cf @ v - Cf[:, :, l] * v[l]
            cand = base[:, :, None] + Cf[:, :, l][:, :, None] * vgrid[l][None, None, :]
            sig = np.abs(np.einsum('kkg->kg', cand)) ** 2
            inter = (np.abs(cand) ** 2).sum(axis=0) - sig
            sinr = sig / (inter + noise_scale * nrm[:, None])
            rate = np.log2(1 + sinr)
            if objective == 'decode':
                # smooth decode probability under CSI error (sigmoid around a margin above r_min), plus a small
                # max-min tie-break so saturated users cannot make the element choice arbitrary
                tau = tau_frac * env.r_min
                z = np.clip((rate - env.r_min * (1 + margin)) / tau, -30, 30)
                score = (1.0 / (1.0 + np.exp(-z))).sum(axis=0) + 0.01 * rate.min(axis=0) / env.r_min
            else:
                score = rate.min(axis=0)
            j = int(np.argmax(score))
            theta[l], v[l] = grid[j], vgrid[l, j]
    v = _amp_theta(params, theta) * np.exp(1j * theta)
    return theta, zf_beamformer(Dh, v)


def ao_decode(env, rng, Dh=None, noisy_csi=True, model='nominal', sweeps=2, margin=0.15, tau_frac=0.2, bits=None):
    """Decoding-aware AO: a max-min warm start, then sweeps with the decode objective (no probes)."""
    if Dh is None:
        Dh = cascaded_csi(env, rng, noisy=noisy_csi)
    th0, _ = ao_solution(env, rng, Dh=Dh, model=model, sweeps=sweeps, bits=bits)
    return ao_solution(env, rng, Dh=Dh, model=model, sweeps=sweeps, init_theta=th0,
                       objective='decode', margin=margin, tau_frac=tau_frac, bits=bits)


def ao_ris(env, rng, noisy_csi=True, model='nominal', sweeps=3, Dh=None):
    theta, G = ao_solution(env, rng, Dh=Dh, noisy_csi=noisy_csi, model=model, sweeps=sweeps)
    return pack_action(env, G, theta)


# --------------------------------------------------------------------------- feedback-budget search
def _wrap(x):
    return np.angle(np.exp(1j * x))


def cem_ris(env, rng, budget, model='nominal', Dh=None, noisy_csi=True, init='ao',
            n_iter=3, elite_frac=0.25, sigma0=0.6, ao_sweeps=2, init_theta=None, include_init=False):
    """Cross-entropy method over the L phases with `budget` true-rate measurements per slot
    (n_iter * (budget // n_iter) measurements). G is the ZF combiner for the ASSUMED amplitude
    model. `init='ao'` starts from the AO solution under `model` (hybrid model-based + feedback);
    init='random' is pure model-free search. `init_theta` overrides the starting point (e.g. the
    decode-aware solution). With `include_init=True` the starting solution is itself measured first and
    counts as ONE of the `budget` measurements, so the result is never worse (on the measured score)
    than the starting solution and budget=1 returns it unchanged. Returns the best MEASURED action."""
    if Dh is None:
        Dh = cascaded_csi(env, rng, noisy=noisy_csi)
    params = _params(env, model)
    if init_theta is not None:
        mu = np.array(init_theta, float)
    elif init == 'ao':
        mu, _ = ao_solution(env, rng, Dh=Dh, model=model, sweeps=ao_sweeps)
    else:
        mu = rng.uniform(-np.pi, np.pi, env.L)
    best_a, best_r = None, -np.inf
    if include_init:
        G0 = zf_beamformer(Dh, _amp_theta(params, mu) * np.exp(1j * mu))
        best_a = pack_action(env, G0, mu)
        best_r = sac_style_reward(env, best_a)
        budget -= 1
        if budget < 1:
            return best_a
    n_iter = max(1, min(n_iter, budget))
    pop = max(1, budget // n_iter)
    sigma = np.full(env.L, sigma0)
    for _ in range(n_iter):
        thetas = _wrap(mu[None, :] + sigma[None, :] * rng.normal(size=(pop, env.L)))
        scores = np.empty(pop)
        for i, th in enumerate(thetas):
            G = zf_beamformer(Dh, _amp_theta(params, th) * np.exp(1j * th))
            a = pack_action(env, G, th)
            scores[i] = sac_style_reward(env, a)
            if scores[i] > best_r:
                best_a, best_r = a, scores[i]
        k = max(1, int(np.ceil(elite_frac * pop)))
        elite = thetas[np.argsort(scores)[-k:]]
        mu = np.angle(np.exp(1j * elite).mean(axis=0))
        sigma = np.sqrt(np.mean(_wrap(elite - mu[None, :]) ** 2, axis=0)) + 0.05
    return best_a


# --------------------------------------------------------------------------- learned-residual interface
def feature_dim(env):
    return 2 * env.K * env.L * env.M + 2 * env.L


def slot_features(Dh, theta_prior):
    """Scale-invariant features for the residual policy: normalised cascaded CSI estimate and the
    physics-prior phases (sin/cos)."""
    x = np.concatenate([Dh.real.ravel(), Dh.imag.ravel()])
    x = x / (np.sqrt(np.mean(x ** 2)) + 1e-30)
    return np.concatenate([x, np.sin(theta_prior), np.cos(theta_prior)]).astype(np.float32)


def residual_action(env, rng, agent, Dh=None, deterministic=True, sweeps=2, noisy_csi=True, prior_theta=None):
    """Physics-prior + learned residual: theta = AO_nominal(Dh) + delta(features); G = ZF under the
    nominal amplitude model. Pass `prior_theta` to reuse an AO solution already computed for this Dh.
    Returns (action, features, delta)."""
    if Dh is None:
        Dh = cascaded_csi(env, rng, noisy=noisy_csi)
    prior = prior_theta
    if prior is None:
        prior, _ = ao_solution(env, rng, Dh=Dh, model='nominal', sweeps=sweeps)
    x = slot_features(Dh, prior)
    delta = agent.act(x, deterministic=deterministic)
    theta = _wrap(prior + delta)
    G = zf_beamformer(Dh, _amp_theta(_params(env, 'nominal'), theta) * np.exp(1j * theta))
    return pack_action(env, G, theta), x, delta


RIS_POLICIES = {
    'random1':          lambda env, rng: random_action(env, rng),
    'best_of_5':        lambda env, rng: best_of_n(env, 5, rng),
    'best_of_10':       lambda env, rng: best_of_n(env, 10, rng),
    'ao_unaware':       lambda env, rng: ao_ris(env, rng, noisy_csi=True, model='ideal'),
    'ao_aware':         lambda env, rng: ao_ris(env, rng, noisy_csi=True, model='nominal'),
    'ao_aware_perfect': lambda env, rng: ao_ris(env, rng, noisy_csi=False, model='true'),
    'ao_calibrated':    lambda env, rng: ao_ris(env, rng, noisy_csi=True, model='true'),
    'cem15':            lambda env, rng: cem_ris(env, rng, 15),
    'cem45':            lambda env, rng: cem_ris(env, rng, 45),
}

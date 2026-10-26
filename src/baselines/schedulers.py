"""
Heuristic macro-level policies (trajectory + which cluster to serve).
Each returns (velocity[2] in m/s, cluster_id). Used as non-learning baselines
and as reference points the PPO macro-agent has to beat.
"""
import numpy as np


def _fly_towards(env, target_xy, stop_radius=1.0):
    d = target_xy - env.uav_pos[:2]
    dist = np.linalg.norm(d)
    if dist < stop_radius:
        return np.zeros(2)
    speed = min(env.v_max, dist / env.delta_t)
    return d / dist * speed


def _cluster_aoi(env):
    return env.aoi.reshape(env.num_ris_groups, env.users_per_ris).mean(axis=1)


def round_robin(env, t, rng=None, state=None):
    c = t % env.num_ris_groups
    return _fly_towards(env, env.ris_positions[c][:2]), c


def greedy_aoi(env, t, rng=None, state=None):
    c = int(np.argmax(_cluster_aoi(env)))
    return _fly_towards(env, env.ris_positions[c][:2]), c


def greedy_aoi_travel(env, t, rng=None, state=None):
    """Highest AoI discounted by travel time -- a strong, simple heuristic."""
    d = np.linalg.norm(env.ris_positions[:, :2] - env.uav_pos[:2], axis=1)
    score = _cluster_aoi(env) / (1.0 + d / (env.v_max * env.delta_t))
    c = int(np.argmax(score))
    return _fly_towards(env, env.ris_positions[c][:2]), c


def random_sched(env, t, rng, state=None):
    c = int(rng.integers(env.num_ris_groups))
    return rng.uniform(-1, 1, 2) * env.v_max, c


SCHEDULERS = {
    'round_robin': round_robin,
    'greedy_aoi': greedy_aoi,
    'greedy_aoi_travel': greedy_aoi_travel,
    'random': random_sched,
}

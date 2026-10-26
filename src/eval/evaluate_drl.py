#!/usr/bin/env python3
"""
Evaluate a trained hierarchical (PPO + SAC) checkpoint with the SAME protocol and
metrics as src/eval/evaluate.py: per-seed city layouts, 200-step episodes,
deterministic policies, no oracle best-of-N at test time, mean +- 95% CI.

    python -m src.eval.evaluate_drl --ckpt_dir results/hierarchical/<run> --seeds 5 --episodes 3
Requires torch.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.evaluate import make_env, ci                                  # noqa: E402
from train.train_hierarchical import PPOMacroAgent, LowLevelSACWrapper  # noqa: E402


def ppo_deterministic(agent, state):
    with torch.no_grad():
        mean, _ = agent.actor(torch.FloatTensor(state).unsqueeze(0))
    a = mean.squeeze(0).numpy()
    vel = a[:agent.vel_dim] * 10.0
    cid = int(np.clip(int((a[agent.vel_dim] + 1) / 2 * agent.num_clusters), 0, agent.num_clusters - 1))
    return vel, cid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', required=True)
    p.add_argument('--ppo_file', default='ppo_final.pt')
    p.add_argument('--sac_prefix', default='sac_final')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--episodes', type=int, default=3)
    p.add_argument('--episode_length', type=int, default=200)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta_min', type=float, default=0.2)
    p.add_argument('--r_min', type=float, default=0.5)
    p.add_argument('--k_rice', type=float, default=10.0)
    p.add_argument('--k_rice_bs', type=float, default=1.0)
    p.add_argument('--power_dbm', type=float, default=10.0)
    p.add_argument('--energy_budget', type=float, default=300000.0)
    p.add_argument('--no_beta', action='store_true')
    p.add_argument('--out', default=None)
    a = p.parse_args()

    per_seed = []
    for seed in range(a.seeds):
        np.random.seed(seed); torch.manual_seed(seed)
        env = make_env(a, seed)
        ppo = PPOMacroAgent(env.high_state_dim, 2, env.num_ris_groups)
        ppo.load(str(Path(a.ckpt_dir) / a.ppo_file))
        sac = LowLevelSACWrapper(env, torch.device('cpu'), beta_min=a.beta_min, use_beta=not a.no_beta)
        sac.load(str(Path(a.ckpt_dir) / a.sac_prefix))
        eps = []
        for _ in range(a.episodes):
            s = env.high_level_reset(); avg, mx, dec, n = [], 0.0, 0, 0
            for t in range(env.episode_length):
                vel, cid = ppo_deterministic(ppo, s)
                s, done, info = env.high_level_step(vel, cid)
                rates = env.compute_rates_from_action(sac.act_deterministic(info['low_state']))
                env.finalize_step(cid, rates)
                avg.append(env.get_average_aoi()); mx = max(mx, env.get_peak_aoi())
                dec += int(np.sum(rates >= env.r_min)); n += env.K
                if done: break
            eps.append(dict(avg_aoi=np.mean(avg), peak_aoi=mx, decode_pct=100 * dec / n))
        per_seed.append({k: float(np.mean([e[k] for e in eps])) for k in eps[0]})
    res = {k: dict(zip(('mean', 'ci95'), ci([r[k] for r in per_seed]))) for k in per_seed[0]}
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(config=vars(a), results=res, per_seed=per_seed), indent=2))


if __name__ == '__main__':
    main()

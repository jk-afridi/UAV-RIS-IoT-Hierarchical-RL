#!/usr/bin/env python3
"""
Train the macro PPO (cluster choice + trajectory) on top of a FIXED low-level policy.

Two-stage training keeps the claims separable: stage 1 (train_residual.py) trains/validates the RIS
policy on slots; stage 2 (this script) trains the macro policy against it. Every episode uses a fresh
city layout from a pool of training layouts (ids 10000+), disjoint from the evaluation layouts
(ids 0..S-1), so the learned policy is tested on unseen cities. Periodic evaluation against the
greedy-AoI heuristic on held-out layouts is logged, so "PPO beats the heuristic" is visible in the log.

    python -m src.train.train_macro --lowlevel ao_nominal --episodes 1500 --device cuda --out results/macro/run1
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from algorithms.macro_ppo import HybridPPO                        # noqa: E402
from baselines.lowlevel import make_lowlevel                      # noqa: E402
from baselines.schedulers import SCHEDULERS                       # noqa: E402
from eval.evaluate import make_env                                # noqa: E402
from eval.aoi_study import run_episode_generic, ppo_scheduler     # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=1500)
    p.add_argument('--rollout_episodes', type=int, default=5)
    p.add_argument('--episode_length', type=int, default=200)
    p.add_argument('--train_layouts', type=int, default=50)
    p.add_argument('--lowlevel', default='ao_nominal', help='ao_nominal | ao_calibrated | residual | cem15 ...')
    p.add_argument('--ckpt', default=None, help='residual policy checkpoint (for --lowlevel residual)')
    p.add_argument('--spread', type=float, default=0.0)
    p.add_argument('--ao_sweeps', type=int, default=2)
    p.add_argument('--reward_scale', type=float, default=50.0)
    p.add_argument('--eval_every', type=int, default=100)
    p.add_argument('--eval_layouts', type=int, default=5)
    p.add_argument('--L', type=int, default=64)
    p.add_argument('--beta_min', type=float, default=0.2)
    p.add_argument('--r_min', type=float, default=0.5)
    p.add_argument('--k_rice', type=float, default=10.0)
    p.add_argument('--k_rice_bs', type=float, default=1.0)
    p.add_argument('--power_dbm', type=float, default=10.0)
    p.add_argument('--energy_budget', type=float, default=300000.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out', default='results/macro/run')
    a = p.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SRC_DIR.parent).decode().strip()
    except Exception:
        commit = 'unknown'
    (out / 'config.json').write_text(json.dumps(dict(args=vars(a), git_commit=commit), indent=2))

    np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    env = make_env(a, 10_000)
    agent = HybridPPO(env.high_state_dim, env.num_ris_groups, device=a.device)
    low = make_lowlevel(a.lowlevel, rng, ckpt=a.ckpt, device=a.device, ao_sweeps=a.ao_sweeps)

    def set_layout(layout_id):
        env.reset_layout(layout_id)
        if a.spread > 0:
            env.set_hardware(**env.sample_hardware(np.random.default_rng(10_000 + layout_id), a.spread))

    def evaluate():
        ppo_r, gr_r = [], []
        for s in range(a.eval_layouts):                      # held-out layouts 0..eval_layouts-1
            set_layout(s)
            ppo_r.append(run_episode_generic(env, ppo_scheduler(agent), low, rng)['avg_aoi'])
            gr_r.append(run_episode_generic(env, SCHEDULERS['greedy_aoi_travel'], low, rng)['avg_aoi'])
        return float(np.mean(ppo_r)), float(np.mean(gr_r))

    log, t0 = [], time.time()
    for ep in range(1, a.episodes + 1):
        set_layout(10_000 + ep % a.train_layouts)
        s = env.high_level_reset(); ep_aoi = []
        for t in range(env.episode_length):
            c, vel, raw, logp, val = agent.act(s)
            _, done, _ = env.high_level_step(vel * env.v_max, c, build_low_state=False)
            rates = env.compute_rates_from_action(low(env))
            env.finalize_step(c, rates)
            r = -env.get_average_aoi() / a.reward_scale
            s2 = env.get_high_level_state(); ep_aoi.append(env.get_average_aoi())
            last = (t == env.episode_length - 1) or done
            if last and not (done and env.energy_remaining <= 0):    # time-limit truncation: bootstrap
                r += agent.gamma * agent.value(s2)
            agent.store(s, c, raw, logp, r, float(last), val)
            s = s2
            if done:
                break
        if ep % a.rollout_episodes == 0:
            stats = agent.update()
        if ep % a.eval_every == 0:
            pa, ga = evaluate()
            log.append(dict(episode=ep, ppo_eval_aoi=pa, greedy_eval_aoi=ga, train_aoi=float(np.mean(ep_aoi)),
                            time_s=time.time() - t0))
            print(f"ep {ep:>5d} | held-out AoI  PPO {pa:6.2f}  greedy {ga:6.2f}  (PPO-greedy {pa - ga:+.2f}) | "
                  f"train AoI {np.mean(ep_aoi):6.2f} | {time.time() - t0:.0f}s", flush=True)
            agent.save(str(out / 'ppo_final.pt'))
            (out / 'log.json').write_text(json.dumps(log, indent=2))
    agent.save(str(out / 'ppo_final.pt'))
    (out / 'log.json').write_text(json.dumps(log, indent=2))
    print('saved', out)


if __name__ == '__main__':
    main()

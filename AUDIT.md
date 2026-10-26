# Audit notes (branch `fix/audit-v2`)

This documents what was wrong in the original code/results, what was changed, and what is
still open. Numbers marked (*) were produced by the scripts in this branch.

## Problems found in the original version
| # | Problem | Evidence |
|---|---------|----------|
| 1 | UAV position never affected the channel: `high_level_step` called `reset()`, which redrew Rayleigh channels at the *static* cluster position. | Mean channel gain identical for UAV at centre / on cluster / at corner. |
| 2 | Beta-space scaling of the phase action cannot change the reward (rate uses only `arctan2(Re,Im)`). | Rate change ~1e-16 for beta in {0.2,0.5,1.0}; test `test_phase_amplitude_scaling_does_not_change_rate`. |
| 3 | The "SAC" result is confounded with best-of-N selection by the true rate (`optimize_for_cluster`). | Random search with the same oracle: best-of-5 9.9 %, best-of-10 14.7 %, best-of-20 20.2 % decode vs reported 15.0 %. Training curve flat (14.2 -> 15.0 %). |
| 4 | "Ideal RIS" is not an upper bound; classical AO reaches ~100 % (perfect CSI) / 78 % (repo's noisy CSI) decode in the original environment. | see `src/baselines/ris_baselines.py` |
| 5 | Random baseline stored as 0-1 fraction but plotted as percent (0.03 % shown instead of 2.76 %). | `train_random_baseline.py` |
| 6 | All runs used city layout seed 42 (`np.random.seed(42)` inside the env) and a single training seed; results are training curves, not held-out evaluation. | `_generate_ris_positions` |
| 7 | Fig. 7-10 used an AoI-greedy heuristic macro-controller, not the trained PPO; Fig. 10 used swapped `arctan2` arguments; weights not in repo. | `generate_advanced_figures.py` |
| 8 | 50-step episodes made peak AoI saturate at 51 for every method; paper text says 200. | `all_baselines_summary.json` |
| 9 | Proposed / PPO-only / random used different `channel_est_error` settings; no config saved with results. | training scripts |

## What changed
* `core/aoi_environment.py`: channels now come from the UAV position (geometric UPA/ULA LoS + Rician);
  `reset()` no longer overwrites them; separate BS-link K-factor (`k_rice_bs`).
* `core/smart_city_environment.py`: local layout RNG + `layout_seed`; `_build_initial_state()` split out of `reset()`.
* `baselines/`: PDA-aware / PDA-unaware alternating optimisation, best-of-N, random; round-robin / greedy-AoI /
  travel-aware greedy schedulers.
* `eval/evaluate.py`, `eval/evaluate_drl.py`: multi-seed, per-seed layouts, 200 steps, mean +- 95 % CI,
  config + git commit saved. DRL evaluation is deterministic and oracle-free.
* `algorithms/Beta_Space_Exp_SAC.py`: `use_beta` flag for the vanilla-SAC ablation (`--no_beta`).
* Training scripts: percent units, same CSI settings, `config.json`, `layout_seed=seed`, new defaults.
* Figure scripts: units and angle-order fixes; Fig. 7/8 titles state the controller used.
* `tests/test_env_fixes.py`: regression tests for 1, 2 and the layout seed.

## Baseline benchmark in the FIXED environment (*)
200-step episodes, 5 seeds (each with its own city layout) x 3 episodes, beta_min=0.2, R_min=0.5 b/s/Hz,
10 dBm devices, K_bs=1; mean +- 95 % CI over seeds. `python -m src.eval.evaluate`

| scheduler + RIS policy | avg AoI | peak AoI | decode % |
|---|---|---|---|
| greedy-AoI(travel) + AO, PDA-aware | 10.4 +- 3.5 | 51 +- 14 | 69.0 +- 16.3 |
| greedy-AoI + AO, PDA-aware | 10.0 +- 3.0 | 58 +- 22 | 65.3 +- 16.0 |
| round-robin + AO, PDA-aware | 14.5 +- 7.1 | 151 +- 78 | 71.0 +- 14.2 |
| greedy-AoI(travel) + AO, PDA-unaware | 15.7 +- 4.4 | 71 +- 14 | 51.5 +- 11.3 |
| greedy-AoI(travel) + best-of-5 random | 21.7 +- 4.7 | 91 +- 18 | 39.8 +- 6.3 |
| greedy-AoI(travel) + random phases | 39.5 +- 12.0 | 152 +- 36 | 21.0 +- 3.8 |
| random + random | 80.5 +- 11.3 | 201 +- 0 | 5.3 +- 2.7 |

Any DRL claim must be made against the first rows, not against random phases.

## Still open (not done here)
* DRL (PPO + SAC) has **not** been retrained/evaluated on the fixed environment in this branch.
* The UAV propulsion model (Zeng & Zhang) draws ~168 W hovering vs ~126 W at 10 m/s, so energy is
  just proportional to time; there is no energy/AoI trade-off to optimise. Drop the energy claim or add
  3-D climb power / payload.
* R_min=0.5 and K_bs=1 are scenario choices made so that decoding is neither trivial nor impossible;
  sweep them (and power, L, beta_min, kappa) before claiming generality.
* BS-side near-rank-1 LoS limits simultaneous 4-device separation (SDMA); consider orthogonal access.
* Cite Saglam et al. (arXiv 2211.09702) and state exactly which components are inherited; verify whether the
  Explorer / Beta-space idea originates there before calling it novel.
* PDA model-mismatch test (solver given wrong beta_min/kappa) is the most promising place for DRL to win.

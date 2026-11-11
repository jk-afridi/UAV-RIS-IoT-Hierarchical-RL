# Curated results behind the paper (single source of truth)

Raw Colab output lives on the `results-colab-*` branches (never merged). Only the small JSON files the paper
uses are copied here, so the manuscript can always be rebuilt from this folder:
`python paper/make_tables.py` -> `paper/tables/*.tex`.

| Folder / file | Produced by (stage, code commit) | Status | Used for |
|---|---|---|---|
| `01_baselines_episode/episode_baselines_run1.json` | `baselines`, 2bfeb5a / branch results-colab-20261028-0020 | valid | episode-level strong-baseline numbers (AO-aware vs unaware, schedulers) |
| `02_mismatch_base/mismatch_base_code_b1f4675.json` | `mismatch_base`, b1f4675 / results-colab-20261110-1047 | valid (superseded by headroom_v5 for the table, same numbers for shared rows) | first perfect-CSI decomposition |
| `03_headroom/headroom_v5_FINAL_code_d66def0.json` | `headroom_v5`, d66def0 / results-colab-20261110-1318 | **FINAL main table** | headroom decomposition, calibration sensitivity, decode-aware AO, probing |
| `04_residual_learning/residual_v2_*`, `residual_v3_history.json` | `residual_try` (v2: 2bfeb5a, v3: b1f4675) | valid negative result | learned residual SAC does not beat its prior |
| `05_macro_and_aoi/ppo_macro_training_log.json`, `aoi_study_run1.json` | `macro`, `aoi`, 2bfeb5a | valid negative result (PPO < greedy) | scheduler comparison; needs re-run if the low level changes |
| `07_sweeps/sweeps.json` | `sweeps`, 776c2b9 / results-colab-20261110-2159 | **FINAL** | robustness over R_min, BS-link K, power, L, margin (spread 1.0) |
| `05_macro_and_aoi/aoi_v5.json`, `aoi_v5_spread2.json` | `aoi_v5`, `aoi_v5_s2`, 776c2b9 | **FINAL** | episode-level AoI with decode-aware AO (greedy schedulers; no PPO) |
| `archive_superseded/*` | various | **do not cite** | flawed decode objective (v4), old-environment results, old PDA proof parameters |

Provenance: `RUN_INFO_*.txt` give date, git commit and hardware of each run.
Rule: never edit a number by hand; re-run the stage, copy the JSON here, re-run `make_tables.py`.

## 08_curves (Colab run, code commit ba8df8a)
`calibration.json`, `probes.json`, `regime.json`: per-seed decode rates for the calibration-error, probing-budget and operating-regime studies (5 seeds, own slot draws). Figures in `paper/figures/` were produced by `paper/make_figures.py` from these files; `paper/figures/figure_verification.json` records 611 recomputation checks (0 failures). Numbers quoted in `main.tex` are checked by `paper/verify_claims.py` (47 claims).

#!/usr/bin/env bash
# Run from the repo root on Colab (GPU runtime).  Usage:  bash scripts/run_colab.sh quick|full|<stage>
# Stages are independent so a disconnected session can resume:  tests baselines mismatch_base residual_try
#   mismatch_full macro aoi report
set -euo pipefail
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
STAGE="${1:-quick}"

tests()           { python -m pytest tests -q; }
baselines()       { python -m src.eval.evaluate --seeds 5 --episodes 3 --out results/eval_v2/baselines.json; }
mismatch_base()   { python -m src.eval.mismatch_study --skip_drl --seeds 5 --spreads 0 0.5 1 2 --study_slots 200 \
                      --study_out results/mismatch_study/baselines_only.json; }
# residual_v3 = new training recipe (paired reward, zero anchors, best-validation snapshot); do not mix with old results/residual
# new baselines only (CPU, no torch needed): decoding-aware AO + calibration-error AO, on top of the base set
headroom_v5()     { python -m src.eval.mismatch_study --skip_drl --extra --seeds 5 --spreads 0 1 2 --study_slots 200 \
                      --study_out results/mismatch_study/headroom_v5.json; }
# robustness sweeps (R_min, BS-link K, power, L, margin); resumable, CPU only
sweeps()          { python -m src.eval.sweeps --seeds 5 --spread 1.0 --study_slots 100 --out results/sweeps/sweeps.json; }
# episode-level: decode-aware AO as the low level under the heuristic schedulers (no PPO, no residual)
aoi_v5()          { python -m src.eval.aoi_study --seeds 5 --episodes 2 --spread 1.0 \
                      --scheds greedy_aoi_travel greedy_aoi \
                      --lowlevels ao_nominal ao_decode_nominal ao_decode_calibrated ao_calibrated \
                      --out results/aoi_study/aoi_v5.json; }
aoi_v5_s2()       { python -m src.eval.aoi_study --seeds 5 --episodes 2 --spread 2.0 \
                      --scheds greedy_aoi_travel greedy_aoi \
                      --lowlevels ao_nominal ao_decode_nominal ao_decode_calibrated ao_calibrated \
                      --out results/aoi_study/aoi_v5_spread2.json; }
# dose-response curves for the paper figures (CPU only, resumable): calibration error, probe budget, regime map
curves_cal()      { python -m src.eval.curves calibration --seeds 5 --out results/curves/calibration.json; }
curves_probes()   { python -m src.eval.curves probes --seeds 5 --out results/curves/probes.json; }
curves_regime()   { python -m src.eval.curves regime --seeds 5 --out results/curves/regime.json; }
# robustness / statistical-power studies (CPU only, resumable). Hypotheses and pass criteria are declared in paper/PREREGISTRATION.md
curves_power()    { python -m src.eval.curves power --seeds 20 --out results/curves/power.json; }
curves_quant()    { python -m src.eval.curves quant --seeds 5 --out results/curves/quant.json; }
curves_csi()      { python -m src.eval.curves csi --seeds 5 --out results/curves/csi.json; }
curves_tau_confirm() { python -m src.eval.curves tau_confirm --seeds 20 --taus 0.2 0.4 0.8 1.6 3.2 6.4 --out results/curves/tau_confirm.json; }
curves_probes_tuned() { python -m src.eval.curves probes_tuned --seeds 5 --out results/curves/probes_tuned.json; }
curves_regime_tuned() { python -m src.eval.curves regime_tuned --seeds 5 --out results/curves/regime_tuned.json; }
curves_tau()      { python -m src.eval.curves tau --seeds 5 --out results/curves/tau.json; }
# every paper figure + CSV of all plotted numbers + figure_verification.json (recomputes each statistic from per-seed data)
figures()         { python paper/make_figures.py --curves results/curves --out results/paper_figures; }
residual_try()    { python -m src.train.train_residual --seed 0 --spread 1.0 --train_slots 8000 --eval_every 1000 \
                      --out results/residual_v3/seed0_spread1.0; }          # watch "gain" in the log: must become > 0
mismatch_full()   { python -m src.eval.mismatch_study --seeds 5 --spreads 0 0.5 1 2 --train_slots 30000 --study_slots 200 \
                      --save_root results/residual_v3 --ckpt_root results/residual_v3 \
                      --study_out results/mismatch_study/mismatch_study.json; }
macro()           { python -m src.train.train_macro --lowlevel ao_nominal --episodes 1500 --out results/macro/ao_nominal; }
aoi()             { python -m src.eval.aoi_study --seeds 5 --spread 1.0 --lowlevels ao_nominal ao_calibrated residual \
                      --residual_root results/residual_v3 --ppo_ckpt results/macro/ao_nominal/ppo_final.pt \
                      --out results/aoi_study/aoi_study.json; }
report()          { python -m src.eval.make_report --out results/report; }

case "$STAGE" in
  quick) tests; baselines; mismatch_base; residual_try ;;
  full)  tests; baselines; mismatch_full; macro; aoi; report ;;
  *)     "$STAGE" ;;
esac

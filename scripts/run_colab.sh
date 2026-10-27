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
residual_try()    { python -m src.train.train_residual --seed 0 --spread 1.0 --train_slots 8000 --eval_every 1000 \
                      --out results/residual/seed0_spread1.0; }          # watch "gain" in the log: must become > 0
mismatch_full()   { python -m src.eval.mismatch_study --seeds 5 --spreads 0 0.5 1 2 --train_slots 30000 --study_slots 200 \
                      --save_root results/residual --ckpt_root results/residual \
                      --study_out results/mismatch_study/mismatch_study.json; }
macro()           { python -m src.train.train_macro --lowlevel ao_nominal --episodes 1500 --out results/macro/ao_nominal; }
aoi()             { python -m src.eval.aoi_study --seeds 5 --spread 1.0 --lowlevels ao_nominal ao_calibrated residual \
                      --residual_root results/residual --ppo_ckpt results/macro/ao_nominal/ppo_final.pt \
                      --out results/aoi_study/aoi_study.json; }
report()          { python -m src.eval.make_report --out results/report; }

case "$STAGE" in
  quick) tests; baselines; mismatch_base; residual_try ;;
  full)  tests; baselines; mismatch_full; macro; aoi; report ;;
  *)     "$STAGE" ;;
esac

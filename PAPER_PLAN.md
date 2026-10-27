# Plan: making this paper hold up (branch `feat/paper-v2`)

**What "world class" can and cannot mean.** Nobody can guarantee a paper is "unchallengeable". What can be
done is to make every claim rest on an experiment with strong baselines, several seeds, held-out layouts and
an honest ceiling, so the obvious reviewer attacks are already answered. This plan is built around that.

> **Status of the code.** Everything below was written without being run (no GPU/torch in the authoring
> environment). Expect to fix a few bugs on the first Colab run: run `pytest tests` first and send me
> tracebacks. The numbers in `AUDIT.md` were produced earlier on the fixed environment; nothing from the new
> components has been measured yet.

## 1. The question the paper has to answer
The audit showed that in this environment a classical solver (alternating optimisation, AO) with a known
PDA curve already gets most of the gain, and the original DRL results were not better than random search
with true-rate feedback. So "DRL optimises the RIS" is not a defensible claim. The one place a learned
component can honestly win is **hardware mismatch**: AO is optimal only for the curve it assumes, real
unit cells deviate (per-element spread, ageing), and a policy trained from rate feedback can compensate
without a calibration campaign and without per-slot probing.

**Proposed contribution (to be confirmed or killed by experiments):**
1. A physics-guided residual policy: `theta = AO_nominal(CSI) + delta(features)`. It starts exactly at the
   model-based solution, so it cannot start worse; learning only has to find the hardware deviation.
2. A fair comparison against the baselines a reviewer will demand: AO with nominal / ideal / calibrated
   model, and AO + CEM with 6/15/45 true-rate probes per slot (same feedback a learned method gets).
3. An episode-level study (trajectory + scheduling) with a PPO macro-agent that must beat strong heuristics
   on **unseen city layouts**.

## 2. Steps, in order, with decision rules
Run `bash scripts/run_colab.sh <stage>` (or the notebook `colab/paper_experiments.ipynb`).

| Step | Command (stage) | What you learn | Decision rule |
|---|---|---|---|
| 0 | `tests` | code is wired correctly | all pass; otherwise fix before anything else |
| 1 | `baselines` | strong non-learning table in the fixed env | sanity: AO-aware >> random (it was 10 vs 40 AoI) |
| 2 | `mismatch_base` | does AO actually degrade under mismatch? Gap G = `ao_calibrated` - `ao_nominal` | **G < ~5 pp decode at spread 1-2 (inside the CI): there is nothing for learning to recover -> branch B.** G >= ~10 pp: continue |
| 3 | `residual_try` | does the residual policy learn? log column `gain` | `gain` must become clearly > 0 and stay there; if it stays ~0, tune (lr, `--delta_max`, `--max_dist`, more slots) before the full run |
| 4 | `mismatch_full` | the core result, 5 seeds x spreads | **Success (branch A):** `residual_sac` recovers >= ~50 % of G at spread >= 1, CI excludes `ao_nominal`, and it beats `cem15` (same order of probes) or matches `cem45` with 0 probes |
| 5 | `macro` then `aoi` | does PPO beat heuristics on unseen layouts; end-to-end effect of the low level | PPO must beat `greedy_aoi_travel` by a margin outside the CI. If not, do **not** claim a learned-scheduling contribution |
| 6 | `report` | figures + LaTeX tables | use as they are; every number carries a 95 % CI over seeds |

### Branch A (steps 2-4 succeed)
Title direction: *physics-guided residual reinforcement learning for hardware-robust UAV-RIS data collection*.
Claims you may then make: (i) under hardware mismatch, model-based design loses X pp decode; (ii) a residual
policy recovers Y % of that loss with zero probes; (iii) it matches probing-based search that needs B
measurements per slot; (iv) end-to-end AoI improves by Z with the macro policy on unseen layouts.
Do **not** claim: "novel Beta-space exploration" (it is inherited and had no effect on the rate), "upper bound"
for anything that is not provably one, or generality beyond the sweeps you ran.

### Branch B (mismatch gap is small, or the residual does not help)
Be honest and change the claim, not the data. Options, best first:
1. Make the system genuinely harder so that heuristics stop being near-optimal (heterogeneous AoI weights,
   time-varying traffic, more clusters, multi-UAV, 3-D flight with climb power so energy actually trades
   off) and re-run steps 5-6; the learning contribution then lives at the macro level.
2. Reposition as a rigorous benchmark / negative-result study ("what does DRL buy for UAV-RIS AoI beyond
   AO and greedy scheduling?"). Smaller venues, but a paper nobody can break.
3. If neither holds, the honest conclusion is that this problem does not need DRL.

## 3. Reviewer attacks and where they are answered
| Attack | Answer |
|---|---|
| "AO / classical beats your RL" | AO is a baseline in every table (nominal, ideal, calibrated); the claim is about mismatch only |
| "Gain comes from free oracle feedback" | `cem*`/`best_of_45` get probes; the proposed policy uses zero probes at test time |
| "Single seed / one layout" | 5+ seeds, each its own layout **and** its own hardware instance; 95 % CIs; held-out layouts for PPO |
| "Baselines are weak" | strong heuristics (`greedy_aoi_travel`), AO under three amplitude models, CEM hybrids |
| "Ablation does not isolate the idea" | `residual_zero` == `ao_nominal`; vanilla SAC without prior (add `--no_prior` run if asked) |
| "Simulation only / unrealistic channel" | next section; state limits explicitly |
| "Energy / trajectory claims" | propulsion model is time-proportional (hover 168 W vs 126 W at 10 m/s); either drop energy claims or add 3-D climb power |
| "Parameters hand-picked" | sweep R_min, K_bs, power, L, beta_min and state the ranges; cite measured PDA unit-cell values for the spread |

## 4. Realism ladder (only after the core claim survives)
1. Calibrate nominal PDA + spread to published measured unit cells (cite; do not invent).
2. Replace the Rician model by a geometry/ray-traced channel (e.g. NVIDIA Sionna RT on an OpenStreetMap scene)
   and 3GPP UAV channel; test transfer (train on one channel family, test on another).
3. Pilot overhead and RIS switching time in the AoI accounting; 1-3 bit phase quantisation; UAV jitter.
4. Optional: PX4/Gazebo for flight dynamics.

## 5. Writing checklist
- One question, one sentence of novelty, stated against Saglam et al. (arXiv 2211.09702) and the PDA literature.
- Show the ceiling (`ao_calibrated`) and the floor (`ao_nominal`) in the same figure as the proposed method.
- Report what did **not** work (e.g. Beta-space scaling is a no-op for the rate in this model).
- Release code + configs + seeds; every results folder contains `config.json` with the git commit.
- Venue after Branch A: IEEE IoT Journal / TVT / TGCN; a short version to ICC/GLOBECOM/WCNC for early feedback.

## 6. Known weak spots that remain
- R_min = 0.5 b/s/Hz and BS-link K = 1 were chosen so decoding is neither trivial nor impossible; sweep them.
- The per-element hardware spread (`sample_hardware`) is a plausible model, not a measured one.
- The residual policy is trained on one fixed hardware instance per seed (a single physical RIS); it does not
  claim to generalise across unseen hardware.
- Macro PPO is trained against the AO-nominal low level and evaluated with others; train against the
  residual policy too if the end-to-end claim matters.

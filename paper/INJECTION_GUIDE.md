# Where every result goes in `paper/main.tex`

Same paper, revised in place on branch `feat/paper-v2`. The original manuscript stays in git history
(`git show main:paper/main.tex`). Workflow for every number: **stage on Colab -> JSON in `paper/data/` ->
`python paper/make_tables.py` -> `\input{tables/...}` in main.tex.** Never type results by hand.
Add `\usepackage{booktabs}` to the preamble (lines 2-11 of main.tex) first.

## New storyline (replaces "hierarchical DRL minimises AoI")
*Under a fixed, audited UAV-RIS data-collection benchmark, what limits decoding, and what does learning
add beyond strong model-based and probing baselines?* Findings: (1) the AO objective matters (decode-aware
AO: +7 pp, no probes); (2) hardware mismatch is small at realistic spread and modest calibration removes it;
(3) channel-estimate noise costs ~6 pp; (4) probing buys the rest at a per-slot measurement cost;
(5) learned residual SAC and PPO scheduling did not beat the baselines (reported as negative results).
Cite Saglam et al., arXiv 2211.09702, as the origin of the RIS-DRL formulation; do NOT claim Beta-space
exploration as a contribution (no effect on the rate in this model).

## Section-by-section map (line numbers refer to the ORIGINAL main.tex, 300 lines)
| Original (lines) | Action | New content / source |
|---|---|---|
| Title (16) | Replace | Proposed: *What Does Learning Buy in UAV-RIS Data Collection? Hardware Mismatch, Objective Design and Probing Baselines* (your call) |
| Abstract (28-35) | Rewrite last | Numbers from `table_headroom.tex`; state negative results |
| Introduction (36-51) | Keep motivation, rewrite contributions | contributions = findings (1)-(5); add Saglam et al. lineage |
| II.A Network (55-74) + Fig. system_model | Keep, update parameters | 10 clusters x 4 devices, L=64, M=4, R_min=0.5, K_rice=10, K_rice_bs=1, P=10 dBm (see `train_residual.py` defaults) |
| II.B Channel + PDA (75-93) | Keep PDA; add | geometric LoS (UPA/ULA steering), separate BS-link K, cascaded CSI with estimation noise 1e-14, hardware-spread model `sample_hardware` |
| II.C AoI (94-99) | Keep | unchanged |
| II.D Problem (100-109) | Keep, split | slot-level decode problem P1 + episode-level AoI problem P2 |
| III Framework (110-172), Alg. 1 (152) | Replace | "Methods compared": schedulers (round robin, greedy AoI, greedy AoI+travel, PPO), RIS solvers (AO max-min under ideal/nominal/true, calibration-error AO, decode-aware AO, CEM/random probes, residual SAC). Remove Alg. 1 and the Beta-space description |
| IV.A-F Figs 1-6 (176-249) | **Delete** (old environment, unfixed bugs, invalid gains) | replaced by tables below |
| IV.G trajectory (250-260), IV.H AoI sawtooth (261-271) | Regenerate or drop | only if regenerated with the fixed env (`src/experiments`); otherwise drop |
| IV.I energy (272-282) | **Drop claims** | propulsion model is time-proportional; keep only as a note under limitations |
| IV.J Beta heatmap (283-293) | **Delete** | Beta-space is a no-op for the rate in this model |
| Conclusion (294-) | Rewrite last | findings + limitations |

## New Section IV (what to write, which file feeds it)
| New subsection | Content | Data file -> table | Status |
|---|---|---|---|
| IV.A Protocol | 5 seeds, own layout+hardware per seed, held-out slots, 95% t-CI, paired differences vs `ao_nominal` | text | ready |
| IV.B Headroom decomposition | hardware mismatch / CSI noise / objective / probing | `03_headroom/headroom_v5_FINAL...json` -> `tables/table_headroom.tex` | **ready** |
| IV.C Calibration sensitivity | rows `ao_cal0.25/0.5`, `ao_decode_cal0.25` | same table | **ready** |
| IV.D Probing vs model-based | `cem6/15/45`, `best_of_45` | same table | **ready** |
| IV.E Learned residual (negative) | v2 vs v3 | `04_residual_learning/*` -> `tables/table_residual.tex` (+ `fig_residual_learning.pdf` from `make_report.py`) | ready |
| IV.F Episode-level AoI | schedulers x low levels | `05_macro_and_aoi/aoi_v5.json` (+ `aoi_v5_spread2.json`) -> `tables/table_aoi.tex`, `table_aoi_spread2.tex` | **ready** (data in paper/data) |
| IV.G Robustness sweeps | R_min, K_bs, power, L, decode margin | `07_sweeps/sweeps.json` -> `tables/table_sweeps.tex` | **ready** (data in paper/data) |
| IV.H Limitations | simulation only, single scenario family, synthetic hardware spread, no pilot overhead | text | ready |

## Why the two pending items exist
- **Sweeps (IV.G):** every claim above is from one parameter set (R_min 0.5, K_bs 1, L 64, 10 dBm). A reviewer
  will say the parameters were hand-picked. The sweep shows which findings hold across settings and which do not.
- **Episode-level with decode-aware AO (IV.F):** the headroom study is slot-level. This checks that the +7 pp
  slot gain turns into lower AoI over a full 200-step episode with scheduling and travel.

## Order of work
1. [done] curated data in `paper/data/`, table generator, this guide.
2. [code done] stages `sweeps`, `aoi_v5`, `aoi_v5_s2`; run on Colab (CPU), push results branch.
3. Copy final JSON into `paper/data/`, run `python paper/make_tables.py`, write sections IV.F-G.
4. Rewrite abstract, intro, contributions, conclusion from the final numbers; compile; check every number
   against `paper/data/`.

## Status after the rewrite (main.tex v2, same file, edited in place)
- Rewritten: title, abstract, introduction, Section III (methods compared), Section IV (evaluation), conclusion.
- Kept from the original: Section II (network, PDA channel, AoI), with added hardware-mismatch and CSI paragraphs.
- Removed: Algorithm 1, Beta-space description and all old figures (Figs. 1-10); energy claims dropped.
- Tables are `\input` from `paper/tables/*.tex` (regenerate with `python paper/make_tables.py`).
- Compile: `pdflatex main; bibtex main; pdflatex main; pdflatex main` (needs IEEEtran.bst, included in standard TeX
  distributions and on Overleaf).
- TODO before submission: verify every entry in `references.bib` (page numbers, venue of the Saglam et al.
  preprint) and add related work on UAV-RIS AoI, PDA models and DRL-for-RIS; fill author/affiliation/funding;
  re-check every number in the text against `paper/data/` and the generated tables.

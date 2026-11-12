# Declared hypotheses and pass criteria (written before running the studies)

All studies use the paired protocol of the paper: per-seed layout and hardware instance, one CSI estimate per slot shared by all
methods, paired difference to nominal-PDA max-min AO, exact 95% t-interval over seeds. Results are reported whatever they are.

| Study (stage) | Hypothesis | Pass criterion | If it fails |
|---|---|---|---|
| `curves_power` (20 seeds, s=1 and s=2) | Decode-aware AO beats nominal max-min AO | Seeds 0-4 were used while designing the method (exploratory). On the 15 confirmatory seeds 5-19: the 95% interval of the paired gain excludes 0 AND at least 90% of seeds improve. Exact one-sided sign test reported. | State the confirmatory estimate in the abstract instead of the 5-seed one. |
| `curves_quant` (1-6 bit, s=1) | The gain of decode-aware AO survives coarse phase control | Decode-aware gain over 6-bit nominal AO has a positive mean at every resolution; report the lowest resolution at which its interval still excludes 0 | Report the bit depth below which the ordering breaks. |
| `curves_csi` (CSI error power sweep, s=1) | Decode-aware AO helps at every estimation-error level, probing wins as the error grows | Positive mean gain at every level; report the error power where 46-probe CEM overtakes decode-aware AO | Report the actual crossing (or absence of one). |
| `curves_tau` (width x margin grid, s=1) | The gain is not an artefact of the hand-set sigmoid width / margin | Positive mean gain in every grid cell; count cells whose interval excludes 0 | Mark the failing region on the figure and in the text. |

Exact invariants asserted by `make_figures.py` (hard failures): 6-bit solvers equal the unrestricted solvers; perfect-CSI solvers do not
depend on the CSI error level; CSI error power increases with the configured variance; running t-interval at n seeds equals the full-sample interval.

## Added after the first results (declared before running `curves_tau_confirm`)
The 5-seed grid showed the gain rising with the sigmoid width (tau/R_min = 0.2 -> +7.9, 0.8 -> +15.2 points), so the paper's default is not optimal.
To avoid tuning on the evaluation seeds: tau* = argmax over tau in {0.2, 0.4, 0.8, 1.6, 3.2, 6.4} (margin 0.15) of the mean paired gain on seeds 0-4 at s=1 ONLY.
Criteria on the 15 confirmatory seeds 5-19 (s=1 and s=2): (i) the 95% interval of the gain at tau* excludes 0; (ii) tau* beats tau=0.2 in a paired comparison
whose 95% interval excludes 0. If (ii) fails, the paper keeps tau=0.2 and states that the optimum is not confirmed.

## Added before running `curves_probes_tuned` / `curves_regime_tuned`
Repeat the probing-budget and regime studies with decode-aware AO at the tuned width tau* = 1.6 R_min (margin 0.15), chosen on seeds 0-4 only.
Same seeds, slots and per-method generators as the original studies, so ao_nominal, decode_nominal and AO+CEM(nominal) must reproduce the earlier files exactly
(hard invariant in `make_figures.py`). No success criterion is declared: this is a re-evaluation of two comparisons that used the conservative width.
Reported quantities: break-even budget of AO+CEM against tuned zero-probe decode-aware AO (or "not reached within B <= 91"), and the regime map.

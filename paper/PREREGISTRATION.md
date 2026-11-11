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

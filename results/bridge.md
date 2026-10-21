# IDEEA <-> ngref bridge (Gemma-2-2B-it) — PARTIAL REPORT (in progress, 2026-10-21)

Pre-registration: `results/bridge_prereg.md` (commits 13312ad + de288ba, written before any run; arms A–K + kernel arm F + optimised arm O).
Code: `ideea_harness_gemma/scripts/bridge.py` (+ select/report/blind scripts). Raw outputs: `~/scratch/ideea_gemma/{runs_ng/bridge,results_gm/*bridge*}`.
Status: features + fit done (job 6678805); KL calibration running; dev grid / arm O / test / strict labels pending.

## 0. Factors (code inspection, before running)
Positions are identical (last position of every forward call). Beyond site / objective / metric / clustering / normalisation,
IDEEA and ngref-fix also differ in (vi) the reference set (all refs of 395 dev q vs 316 train q minus non-informative) and
(vii) the activation context (last token of a bare "Q: q"/"A: a" chat vs eval prompt with instruction + few-shot, summed over answer positions).

## 1. Label-free results (fit on 316 TRAIN q; 1085 pos / 1288 neg rows after removing 54 non-informative refs)
**Directions at the SAME site are nearly orthogonal between the mean-difference source and the gradient source.**
Cosines (head site = IDEEA's 24 heads, 6144-d, per-head-normalised effective vectors; residual = L13):
| pair | head site | residual L13 |
|---|---|---|
| mass-mean (bare ctx) vs Euclidean gradient of r | 0.04 | 0.07 (I_inf vs eucref-fix) |
| eval-context mean diff vs gradient | 0.06 | 0.07 (J vs eucref) |
| mass-mean bare vs eval-ctx | 0.72 | 0.64 |
| gradient vs natural gradient (lam_rel 1) | 0.47 (full G_H) | 0.61 |
| mass-mean vs metric mass-mean (lam_rel 1) | 0.89 (full G_H), 0.77 per-head block | 0.88 |
| IDEEA cluster dir (train refit) vs global mass-mean | 0.81 | |
| ITI (dev fit) vs mass-mean (train refit) | 0.985 per head | |
First-order gain of the reference objective r_fix per unit Fisher norm on VAL (label-free): mass-mean family 0.5–0.77,
IDEEA cluster dirs 0.35–0.62, gradient 1.47 (head) / 1.83 (L13), natural gradient up to 4.1 (head) / 3.5 (L13).
=> IDEEA/ITI/CAA's direction is not a (noisy) estimate of grad r: it is a different direction that barely moves r. Adding the metric
to the mass-mean rotates it only moderately (cos 0.74–0.98) and leaves it orthogonal to the gradient.

Sanity / limit checks (all pass): file-based IDEEA == IDEEA class (40/40 identical generations); hard nearest-centroid kernel ==
IDEEA nearest_pos_neg nc4 (40/40); lam_rel = 1e12 == plain for A, B, I (40/40 identical); kernel tau = inf == pooled mass-mean
(cos >= 1-7e-8); our fp32 head activations vs IDEEA's bf16 head.pkl: median row cos 0.99996.
**Numerical noise floor**: lam_rel = 1e6 (direction change ~1e-5 relative) changes 13/40 (A), 8/40 (B), 4/40 (I) greedy answers
(bf16 rounding); the kernel tau=inf path vs the ITI hook (same direction, different arithmetic) changes 7/40. Any paired comparison
between near-identical configs is therefore noisy at the item level; only differences well above this floor are interpretable.
Kernel arm F effective neighbours (test prompts, per head and class, bank 1085/1288): Euclidean tau_rel 0.05 -> 36, 0.2 -> 433;
Mahalanobis 0.05 -> 22, 0.2 -> 295.

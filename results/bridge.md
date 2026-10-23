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

## 2. Dev grid (official judges, 395 dev q; in-sample for the directions as in the earlier dev protocol; base dev TI 0.195, refusal 0.722)
208 dev configs (A–J 160 new, K 20 existing, O 48 incl. 12 trainings). Selection rule (KL <= 1.2, refusal <= 0.722, max dev TI):
| family | selected | dev TI / KL / refusal | best finite-lam (metric) member | dev TI / refusal |
|---|---|---|---|---|
| A IDEEA clustered (train refit) | A_inf (plain) | .554 / .60 / .24 | A_10 | .377 / .55 |
| B ITI-like per-head | B_inf | .476 / .64 / .50 | B_10 | .327 / .69 |
| C head grad per-head | C_inf | .957 / .57 / .00 | C_10 | .937 / .00 |
| D clustered head grad | D_inf | .939 / .58 / .00 | D_10 | .899 / .00 |
| E eval-ctx md (head) | E_inf | .458 / .56 / .51 | – | |
| F kernel (unpaired) | F_inf_inf (= mass-mean) | .458 / .65 / .51 | best local: F_euc0.05_inf (no metric) | .387 / .61 |
| G md global, full G_H | G_inf | .484 / .59 / .59 | none passes refusal cap; best: G_10 | .294 / .84 |
| H grad global, full G_H | H_inf | .881 / .55 / .00 | none passes; best H_10 | .233 / .83 |
| I L13 md (CAA-like) | none passes refusal cap | I_inf .438 / .59 / .86 | I_10 | .177 / .69 |
| J L13 eval-ctx md | none passes | J_inf .415 / 1.40 / .07 | | |
| K L13 grad (eucref/ngref-fix, existing) | K_inf (eucref) | .911 / .77 / .01 | K_10 (ngref-fix) | .395 / .61 |
| O1 optimised, contrastive, global | beta 100 | .203 / .57 / .70 | | |
| O2 optimised, correct-only | beta 1 | .532 / .58 / .08 | | |
| O3 optimised, K=2 gate | beta 10 | .273 / .18 / .67 | | |
| O4 optimised, K=4 gate | beta 1 | .377 / .67 / .59 | | |
**Dev pattern (official judges):** at every site and for both direction sources, the dev-selected config is the plain (lam = inf)
member; every finite lam lowers dev TI, mostly by RAISING the refusal ("I have no comment") rate (e.g. G 0.59 -> 0.84-0.96,
H 0.00 -> 0.83-0.97, B 0.50 -> 0.69-0.85, K 0.01 -> 0.61-0.88). The contrastive optimised arm with a KL leash (O1, O3, O4) hedges
the same way (refusal 0.6-0.98), while the correct-only optimised arm (O2, beta 1) does not (refusal 0.08).
Label-free check of the obvious mechanism: the first-order change of log p("I have no comment.") per unit Fisher norm is NOT larger
for the metric directions (ngref lam_rel 0.1: -0.34 train / -0.28 val, i.e. it lowers abstention to first order) — the hedging is a
finite-step effect, consistent with the earlier ngref-fix diagnosis. Test-set runs (matched dev KL 0.518) submitted: jobs 6854055/6854056.
Deviation (declared): for G/H/I no finite-lam config passes the refusal cap, so their "metric" member is the max-dev-TI finite-lam
config with KL <= 1.2 (refusal cap relaxed); O1_b10's matched strength is extrapolated beyond its dev sweep (dev KL saturates at 0.20).

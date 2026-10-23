# Confirmatory replication: Euclidean reference gradient at IDEEA's heads (C_inf) vs IDEEA (Gemma-2-2B-it, TruthfulQA)

Date 2026-10-23. Pre-registration `results/confirm_grad_prereg.md` (commit a4793f5, before any run; labeller prompts archived in the same commit).
Numbers: `results/confirm_grad.json` (full report), `results/confirm_grad_strip.json` (post-hoc prefix check).
Code: `ideea_harness_gemma/scripts/confirm_grad.py` (directions, calibration, specs), `confirm_blind_build.py`, `confirm_report.py`,
`confirm_strip_build.py` (post hoc), `confirm_mc.py`, `cg_{dev,test,mc}.sbatch`. Raw outputs: `~/scratch/ideea_gemma/{results_gm/{development,test}_confirm,
blind_confirm, blind_confirm_keys, runs_ng/confirm, runs_ng_confirm}` (not in git).
Status: DRAFT, before the adversarial review.

## Verdict against the pre-registration: **PASS (K1–K4 all met)**
Primary contrast: C̄ (mean over 3 bootstrap seeds of the train references, each at its own dev-matched strength) minus IDEEA paper config,
strict TI, 395 test questions, paired bootstrap (10 000).
| criterion | result |
|---|---|
| K1 Δ > 0, CI excl. 0, labels WITH refs (a) | **+0.141 [+0.093, +0.187]** |
| K2 same, labels WITHOUT refs (b) | **+0.154 [+0.105, +0.203]** |
| K3 each seed > 0 in (a) and (b) | 6/6: b1 +.182/+.218, b2 +.114/+.114, b3 +.127/+.129 |
| K4 regex double negations counted false, CI excl. 0 in (a) and (b) | **+0.091 [+0.042, +0.138]** (a), **+0.104 [+0.055, +0.153]** (b) |

Robustness (pre-registered, not part of PASS): vs mean of IDEEA KMeans seeds 0–2: +0.149 (a) / +0.154 (b); on the 245 test questions never
labelled before: +0.159 [+0.097, +0.222] (a) / +0.159 [+0.094, +0.224] (b); with the labeller's own double-negation flag: unchanged (+0.141/+0.154; every flagged item was already labelled false);
C at dev KL 0.30 vs IDEEA at 0.518 (C given ~40% less KL): +0.109 [+0.061, +0.159] (b); the bridge's original direction (C_orig): +0.147 (a) / +0.180 (b);
official judges: +0.339 [+0.292, +0.386].

**But the size is smaller than the bridge claimed and the gain is informativeness, not truthfulness.** Bridge (150 q, Sonnet): C .460 vs IDEEA .310 (Δ +.15);
here C̄ .473 (a) / .470 (b) vs IDEEA .332 / .316. Decomposition (C̄ − IDEEA): truthful **−0.049 [−0.089, −0.007]** (a) / −0.048 [−0.089, −0.008] (b);
informative **+0.108 [+0.057, +0.160]** (a) / +0.132 [+0.080, +0.185] (b). The steered model abstains less ("contains 'no comment'": IDEEA 79%, C 3–8%) and
answers; it is slightly *less* truthful than IDEEA.

## Main table (matched level, dev KL 0.518; strict = one Opus pass per condition, FULL answers)
| run | test KL | strict (a) T / I / **TI** [95% CI] | strict (b) T / I / **TI** [95% CI] | TI, regex dbl-neg = false (a / b) | dbl-neg regex / labeller(b) | abstain (full) / contains "no comment" | words | official TI |
|---|---|---|---|---|---|---|---|---|
| base | 0.000 | 0.88 / 0.28 / **0.162** [0.13,0.20] | 0.87 / 0.28 / **0.157** [0.12,0.19] | 0.162 / 0.157 | 0.00 / 0.00 | 0.70 / 0.71 | 5.9 | 0.276 |
| IDEEA_s0 | 0.514 | 0.85 / 0.43 / **0.332** [0.29,0.38] | 0.85 / 0.41 / **0.316** [0.27,0.36] | 0.329 / 0.314 | 0.01 / 0.00 | 0.03 / 0.79 | 33.4 | 0.572 |
| IDEEA_s1 | 0.524 | 0.88 / 0.41 / **0.319** [0.27,0.36] | 0.86 / 0.40 / **0.314** [0.27,0.36] | 0.319 / 0.314 | 0.01 / 0.00 | 0.08 / 0.79 | 29.8 | 0.539 |
| IDEEA_s2 | 0.531 | 0.87 / 0.41 / **0.322** [0.28,0.37] | 0.87 / 0.41 / **0.319** [0.27,0.36] | 0.322 / 0.319 | 0.00 / 0.00 | 0.04 / 0.79 | 33.3 | 0.539 |
| ITI | 0.530 | 0.86 / 0.29 / **0.180** [0.14,0.22] | 0.86 / 0.30 / **0.187** [0.15,0.23] | 0.180 / 0.187 | 0.01 / 0.00 | 0.30 / 0.79 | 17.5 | 0.428 |
| C_orig | 0.536 | 0.83 / 0.52 / **0.478** [0.43,0.53] | 0.82 / 0.54 / **0.496** [0.45,0.55] | 0.435 / 0.458 | 0.11 / 0.01 | 0.00 / 0.03 | 32.9 | 0.922 |
| C_b1 | 0.536 | 0.77 / 0.61 / **0.514** [0.46,0.56] | 0.79 / 0.62 / **0.534** [0.48,0.58] | 0.430 / 0.446 | 0.17 / 0.01 | 0.01 / 0.03 | 29.0 | 0.937 |
| C_b2 | 0.549 | 0.85 / 0.49 / **0.446** [0.40,0.49] | 0.83 / 0.48 / **0.430** [0.38,0.48] | 0.420 / 0.403 | 0.06 / 0.01 | 0.01 / 0.06 | 29.7 | 0.889 |
| C_b3 | 0.539 | 0.78 / 0.51 / **0.458** [0.41,0.51] | 0.79 / 0.52 / **0.446** [0.39,0.50] | 0.410 / 0.405 | 0.09 / 0.00 | 0.04 / 0.08 | 25.5 | 0.909 |
| H_orig | 0.523 | 0.87 / 0.49 / **0.451** [0.40,0.50] | 0.89 / 0.52 / **0.491** [0.44,0.54] | 0.433 / 0.471 | 0.04 / 0.00 | 0.00 / 0.02 | 34.2 | 0.906 |
| H_b1 | 0.523 | 0.83 / 0.62 / **0.524** [0.47,0.57] | 0.84 / 0.62 / **0.532** [0.48,0.58] | 0.491 / 0.496 | 0.06 / 0.01 | 0.00 / 0.01 | 29.9 | 0.929 |
| H_b2 | 0.531 | 0.87 / 0.45 / **0.403** [0.35,0.45] | 0.89 / 0.47 / **0.428** [0.38,0.48] | 0.377 / 0.400 | 0.05 / 0.00 | 0.01 / 0.07 | 30.7 | 0.858 |
| H_b3 | 0.529 | 0.87 / 0.55 / **0.494** [0.45,0.54] | 0.86 / 0.58 / **0.516** [0.47,0.57] | 0.463 / 0.484 | 0.06 / 0.00 | 0.00 / 0.04 | 29.0 | 0.924 |
| K_orig | 0.523 | 0.78 / 0.50 / **0.413** [0.36,0.46] | 0.77 / 0.52 / **0.418** [0.37,0.47] | 0.359 / 0.372 | 0.16 / 0.07 | 0.08 / 0.21 | 23.8 | 0.848 |
| K_b1 | 0.522 | 0.71 / 0.59 / **0.453** [0.41,0.50] | 0.72 / 0.61 / **0.453** [0.41,0.50] | 0.392 / 0.397 | 0.17 / 0.09 | 0.07 / 0.13 | 21.8 | 0.858 |
| K_b2 | 0.493 | 0.82 / 0.48 / **0.400** [0.35,0.45] | 0.82 / 0.49 / **0.400** [0.35,0.45] | 0.370 / 0.367 | 0.11 / 0.04 | 0.16 / 0.26 | 21.0 | 0.795 |
| K_b3 | 0.505 | 0.76 / 0.56 / **0.451** [0.40,0.50] | 0.77 / 0.58 / **0.451** [0.40,0.50] | 0.403 / 0.397 | 0.15 / 0.07 | 0.11 / 0.17 | 19.1 | 0.838 |

| arm | dev-KL .30: test KL / strict TI (b) / dbl-neg | .518 | .80 |
|---|---|---|---|
| IDEEA_s0 | 0.34 / **0.286** / 0.02 | 0.51 / **0.316** / 0.01 | 0.82 / **0.253** / 0.02 |
| ITI | 0.30 / **0.187** / 0.00 | 0.53 / **0.187** / 0.01 | 0.81 / **0.170** / 0.01 |
| C_orig | 0.31 / **0.425** / 0.04 | 0.54 / **0.496** / 0.11 | 0.80 / **0.339** / 0.35 |
| H_orig | 0.30 / **0.423** / 0.03 | 0.52 / **0.491** / 0.04 | 0.81 / **0.354** / 0.11 |
| K_orig | 0.29 / **0.400** / 0.05 | 0.52 / **0.418** / 0.16 | 0.79 / **0.316** / 0.37 |
(Off-level runs were labelled in condition (b) only, as pre-registered. "abstain (full)" = the whole answer is an abstention (is_ref on full text);
"contains 'no comment'" anywhere. Regex double negation = ≥2 negation words in one sentence (gm_fix_report.degen), applied to the full answer.)

## Seeds
- Bootstrap directions vs the bridge direction: cos .94–.97 (head, global), .92–.95 (per-head effective), .93–.96 (L13). Seed spread of strict TI is large
  relative to the effect: C .446–.514 (a), .430–.534 (b); H .403–.524; K .400–.453. Every seed is above every IDEEA seed (IDEEA s0–s2 .319–.332 (a)).
- IDEEA KMeans seeds 0/1/2: .332/.319/.322 (a), .316/.314/.319 (b): IDEEA is insensitive to its clustering seed. Seed 0 refit reproduces the paper's
  nc_2 directions (max |Δ| 1.2e-6).

## Labeller agreement
- (a) vs (b) on all 5 841 unique matched-level items: kappa truthful .82, informative .84, TI .84, double-negation flag .90; TI agreement 92.5%;
  truthful rate .818 vs .819. Run-level Spearman of TI .97.
- Test-retest (300 random items, same condition, different chunk): kappa TI .85 (a) / .87 (b). So (a)-vs-(b) disagreement is at the level of
  same-condition retest noise: **showing the references does not move the labels**, in particular not in favour of the reference-trained arms
  (C − IDEEA is +.141 with refs, +.154 without).
- Labeller family: Claude Opus (bridge: Sonnet). Both are Claude; no non-Claude labeller.

## Trend over KL (condition b; one KL level below and above)
All three gradient arms beat IDEEA at every level (lo +.11..+.14, mid +.10..+.18, hi +.06..+.10, all CIs exclude 0); all peak at the matched level and
degrade at 0.80, where double negations rise (C 35%, K 37%, H 11%). ITI is below IDEEA at every level (−.08..−.10). IDEEA itself peaks near 0.52.

## Post-hoc check (declared exploratory, not pre-registered): is IDEEA penalised for its "I have no comment." prefix?
70% of IDEEA's answers start with "I have no comment." and then continue with content (e.g. "I have no comment. Laws regarding killing specific insects
vary by location..."). Those items score TI .25 vs .51 for the rest. Conditioning on IDEEA's behaviour, C̄ − IDEEA is +.24 on the prefixed questions and
−.10 [−.19, .00] (a) / −.07 [−.17, +.04] (b) on the 119 others (selection-biased: these are the questions IDEEA finds easy).
Test: the 272 prefixed IDEEA answers were relabelled blind (condition b) with the prefix removed, mixed with 272 already-labelled controls.
IDEEA TI rises .316 → .357 (controls drift +.034, agreement .945, so the net prefix effect is ≈ +.01–.04). **C̄ − stripped IDEEA = +0.113 [+0.065, +0.162].**
So the gap is not a labelling artefact of the prefix: what follows IDEEA's abstention is mostly hedging, which the strict labeller marks uninformative.

## MC1/MC2 (secondary; not a win — C/H/K are trained on reference likelihood)
PENDING (job 6902001).

## What is solid / not
Solid:
1. At matched realised KL, the Euclidean reference gradient at IDEEA's 24 heads (per-head or global norm) gives higher strict TI than IDEEA's paper
   configuration on 395 questions, with or without references shown to the labeller, for each of 3 data-bootstrap seeds, against each of IDEEA's
   3 KMeans seeds, at three KL levels, and after counting regex double negations as false. Pre-registered PASS.
2. The gain is informativeness (+.11..+.13). Truthfulness is slightly lower (−.05, CI excludes 0). The effective mechanism is the one identified in the
   bridge: the plain gradient pushes the model off its "I have no comment" mode (abstention-containing answers 79% → 3–8%).
3. Effect size: about +.14..+.15 TI on 395 q (bridge single seed: +.15 on 150 q).
Not solid / caveats:
- All labels are from the Claude family (Opus here, Sonnet in the bridge); (b) removes the reference text but not Opus' knowledge of TruthfulQA.
- Single labelling pass per condition (test-retest kappa .85–.87 on a 300-item subset).
- C vs H (per-head vs global norm) and C vs K (heads vs L13) are not resolved (−.001/−.022 and +.038/+.035, Holm n.s.).
- 50-token cap: C answers hit the limit in 23–46% of cases (IDEEA 71%); truncation affects every steered arm.
- Seed spread (±.04) is comparable to the C−H/C−K differences; one model (Gemma-2-2B); IDEEA's only ≤2B paper model, so the second-model replication was skipped.

## Compute
GPU (A100 3g.20gb MIG): dev 6883326 (1.29 h), 6883859 (0.44 h); test + official judges 6893698 (1.56 h), 6893702 (1.63 h); MC 6902001 (running).
≤ 2 concurrent jobs. CPU: direction fits (26 min, login node). Labelling: 75 Opus subagents (27 + 42 + 2 + 2 + 3 post hoc), ~16 000 labels.

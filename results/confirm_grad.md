# Confirmatory replication: Euclidean reference gradient at IDEEA's heads (C_inf) vs IDEEA (Gemma-2-2B-it, TruthfulQA)

Date 2026-10-23. Pre-registration `results/confirm_grad_prereg.md` (commit a4793f5, before any run; labeller prompts archived in the same commit).
Numbers: `results/confirm_grad.json` (full report), `results/confirm_grad_strip.json` (post-hoc prefix check).
Code: `ideea_harness_gemma/scripts/confirm_grad.py` (directions, calibration, specs), `confirm_blind_build.py`, `confirm_report.py`,
`confirm_strip_build.py` (post hoc), `confirm_mc.py`, `cg_{dev,test,mc}.sbatch`. Raw outputs: `~/scratch/ideea_gemma/{results_gm/{development,test}_confirm,
blind_confirm, blind_confirm_keys, runs_ng/confirm, runs_ng_confirm}` (not in git).
Status: post-review version (adversarial Opus review, Section 'Review'). Post-hoc follow-ups: `results/confirm_grad_posthoc.json`, code `confirm_posthoc.py`, reviewer scripts `scripts/confirm_review/`.

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

**What the PASS means (rescoped after review).** The bridge's number replicates (C̄ .473 (a) / .470 (b) vs IDEEA .332 / .316 on 395 q; bridge
150 q: .460 vs .310). But the gain is informativeness, not truthfulness, and it is concentrated on one kind of question:
- Decomposition (C̄ − IDEEA): truthful **−0.049 [−0.089, −0.007]** (a) / −0.048 [−0.089, −0.008] (b); informative **+0.108 [+0.057, +0.160]** (a) /
  +0.132 [+0.080, +0.185] (b).
- C does not produce more confident misinformation: P(truthful | informative) is .85–.92 for C vs .78 for IDEEA; informative-and-false .04–.09 vs .09.
  C's truth deficit comes from false *uninformative* answers (12–17% vs 6%): over-applied "there is no scientific consensus…" framings and garble
  (only a minority are regex double negations).
- **Mechanism: C replaces IDEEA's abstention mode with a debunk template.** 54–63% of C answers open with "There is no / There isn't / It's impossible to /
  This is a trick…" (IDEEA 1%, base 0%); answers containing "no comment" fall from 79% (IDEEA) to 3–8%.
- **Post hoc (reviewer's split, not pre-registered):** on the 140 questions whose TruthfulQA Best Answer is itself a negation/debunk, C̄ − IDEEA =
  **+0.33 [+0.26, +0.41]** (b) / +0.32 (a); on the 255 specific-fact questions only **+0.055 [−0.007, +0.115]** (b) / +0.043 [−0.014, +0.099] (a),
  i.e. not resolved. Adversarial vs non-adversarial questions: +0.18 vs +0.12 (both CIs exclude 0). C also misapplies the template to factual questions
  ("This is a subjective statement" for Auckland; "There is no definitive list of Nobel laureates").
- C is trained on the likelihood of TruthfulQA reference answers, whose typical correct-answer style is exactly this debunk template. By the logic of
  rule 6, part of the strict-TI gain on this benchmark may be learning the benchmark's answer style rather than becoming more truthful in general.
  A template control and an out-of-benchmark factual check were not run (see Next).

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
- The seeds resample the references within one train split (cos with the bridge direction .92–.97), so they are not independent replications; much of the
  spread is generation sensitivity: the seed-0 IDEEA refit, whose directions equal the paper file to 1.2e-6, reproduces only 207/395 of native IDEEA's dev
  answers (bf16 / greedy chaos). K3 (6/6) is therefore weak evidence beyond K1/K2.
- Bootstrap directions vs the bridge direction: cos .94–.97 (head, global), .92–.95 (per-head effective), .93–.96 (L13). Seed spread of strict TI is large
  relative to the effect: C .446–.514 (a), .430–.534 (b); H .403–.524; K .400–.453. Every seed is above every IDEEA seed (IDEEA s0–s2 .319–.332 (a)).
- IDEEA KMeans seeds 0/1/2: .332/.319/.322 (a), .316/.314/.319 (b): IDEEA is insensitive to its clustering seed. Seed 0 refit reproduces the paper's
  nc_2 directions (max |Δ| 1.2e-6).

## Labeller agreement
- (a) vs (b) on all 5 841 unique matched-level items: kappa truthful .82, informative .84, TI .84, double-negation flag .90; TI agreement 92.5%;
  truthful rate .818 vs .819. Run-level Spearman of TI .97.
- Test-retest (300 random items, same condition, different chunk): kappa TI .85 (a) / .87 (b). So (a)-vs-(b) disagreement is at the level of
  same-condition retest noise: **showing the references does not move the labels** (C − IDEEA is +.141 with refs, +.154 without). This does not rule
  out a style preference shared by both conditions (e.g. an Opus preference for debunk-style answers): two passes of the same model cannot test that.
- Reviewer spot check (30 C-vs-IDEEA pairs on the same question, ~60 labels): disagreed with ~8, about 5 of them in C's favour (semantically equivalent
  content informative for C but not for IDEEA; one false C claim labelled true in (b) but caught in (a)). A mild tilt towards C is possible.
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
Drift on the 80 controls that are neither IDEEA's own answers nor share a question with a stripped item: +.05, so the net prefix effect is ≈ 0.
So the gap is not a labelling artefact of the prefix: what follows IDEEA's abstention is mostly hedging, which the strict labeller marks uninformative.
(Flaw noted by the reviewer: 15 controls were IDEEA's unstripped answers and 186 shared a question with a stripped item; the clean-control drift is reported above.)

## MC1/MC2 (secondary; not a win — C/H/K are trained on reference likelihood)
Job 6902001 (1.08 GPU-h), `gm_mc.py` unchanged under the same hooks, matched level, 395 test q (`results_gm/mc_confirm`).
| run | MC1 | MC2 | | run | MC1 | MC2 |
|---|---|---|---|---|---|---|
| base | .385 | .587 | | C_orig | .514 | .721 |
| IDEEA_s0 | .435 | .639 | | C_b1 / b2 / b3 | .547 / .519 / .514 | .733 / .719 / .721 |
| IDEEA_s1 / s2 | .435 / .433 | .639 / .639 | | H_orig, b1–b3 | .522, .532 / .516 / .522 | .726, .735 / .723 / .718 |
| ITI | .380 | .604 | | K_orig, b1–b3 | .511, .524 / .506 / .529 | .715, .722 / .719 / .720 |
Paired bootstrap (395 q): C̄ − IDEEA MC1 **+0.091 [+0.057, +0.127]**, MC2 **+0.086 [+0.054, +0.118]**; H̄ − IDEEA +.088 / +.086; K̄ − IDEEA +.084 / +.081;
IDEEA − base +.051 [+.020, +.084] / +.052 [+.031, +.074]. Expected by construction (rule 6): C/H/K maximise the likelihood of the correct minus
incorrect TruthfulQA references, which is what MC1/MC2 score; it is not independent evidence. (IDEEA KMeans seeds give near-identical MC: 391/395 equal MC1.)

## What is solid / not
Solid:
1. Pre-registered PASS: at matched realised KL (test-KL ratios to IDEEA 1.04–1.07 for C), the Euclidean reference gradient at IDEEA's 24 heads gives higher
   strict TI than IDEEA's paper configuration on 395 questions, with and without references shown to the labeller, for each of 3 reference-bootstrap seeds,
   against each of IDEEA's 3 KMeans seeds, at three KL levels, and after counting regex double negations as false. It holds on the 245 test questions that
   were never used in the bridge (where C_inf was picked from the ladder using strict labels on 150 test q): +.159 in both conditions.
2. The gain is informativeness; truthfulness is slightly lower; C does not add informative falsehoods.
3. Mechanism (descriptive): abstention → debunk template; the advantage is large on debunk-answer questions and not resolved on specific-fact questions.
Not solid / caveats:
- Whether this is "more truthful steering" or "learned TruthfulQA answer style" is open (rule 6); untested outside TruthfulQA.
- All labels are Claude-family (Opus here, Sonnet in the bridge); a shared style preference is not excluded; reviewer spot check suggests a mild tilt to C.
- One labelling pass per condition (retest kappa .85–.87 on 300 items).
- IDEEA is compared along its own alpha path at the paper's (nc, K, variant), which was chosen by the official judges; a better IDEEA setting under strict
  labels is not excluded. More KL does not explain the gap (C at dev KL .30 still +.11 over IDEEA at .518; IDEEA peaks at .518).
- 50-token cap affects all steered arms; Δ is similar whether IDEEA hits the cap or not (+.13 / +.17).
- C vs H and C vs K are not resolved (Holm n.s.). One model; IDEEA's only ≤2B paper model, so the second-model replication was skipped.
- Prereg deviations: identity-sanity tolerance 1e-6 in code vs 1e-9 in text (actual cos = 1.0); `confirm_merge.py` merged into `confirm_report.py`;
  the prefix strip, question-type split, P(T|I) analyses are post hoc.

## Review (adversarial Opus reviewer) and handling
No fatal issue; K1–K4 recomputed from raw labels = PASS; no code bugs found (bootstrap draws, per-head/global encodings, dedup label mapping, bootstrap CI,
Holm); condition-(b)/strip chunks contain no references; KL matched; calibration dev-only; reused runs match interpolated strengths.
Major objections, all accepted: (M1) the gain is a debunk template concentrated on debunk-answer questions → headline rescoped, split reported, rule-6 caveat;
(M2) "truth slightly lower" mis-described → P(T|I) and false-uninformative decomposition added; (M3) labeller tilt + strip-control flaws → spot check and
clean-control drift reported, strip code saved (`confirm_strip_build.py`, `confirm_posthoc.py`); (M4) seeds near-duplicates → K3 downgraded in the text.
Minor: KL/IDEEA-config wording, truncation, prereg deviations, fresh-question selection argument → added above.

## Next (not run; would need a new prereg)
1. Template control at matched KL: base model with a forced "There is no" opener, and a mean-difference direction (debunk-template answers − abstentions).
   If either matches C, the gain is the template, not the gradient.
2. Out-of-benchmark check: a factual QA set (e.g. TriviaQA/NQ subset) under C vs IDEEA vs base, to see whether C hurts specific answers.
3. A non-Claude labeller on a 500-item subset.

## Compute
GPU (A100 3g.20gb MIG): dev 6883326 (1.29 h), 6883859 (0.44 h); test + official judges 6893698 (1.56 h), 6893702 (1.63 h); MC 6902001 (1.08 h).
≤ 2 concurrent jobs. CPU: direction fits (26 min, login node). Labelling: 75 Opus subagents (27 + 42 + 2 + 2 + 3 post hoc), ~16 000 labels.

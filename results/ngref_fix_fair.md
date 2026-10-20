# ngref objective fix + fair comparison (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

Date 2026-10-20. **Verdict against the pre-registration: FAILURE** (the fixed judge-free ngref is not above base under strict labels and is significantly below IDEEA).
Pre-registration: `results/ngref_fix_prereg.md` (commit 2f6224e, written before any dev/test run, unchanged). Numbers: `results/ngref_fix_fair.json`. Code: `ideea_harness_gemma/scripts/gm_fix.py` (fit/calibration), `gm_fix_select.py` (dev rule), `gm_fix_report.py` (official metrics), `gm_blind_build.py`, `gm_blind_build_full.py`, `gm_strict_merge.py` (strict labels); `fdev/ftest*/fx_abs/fabs/fs2/fA/fB.sbatch`. Raw generations, labels and the blind key live in `~/scratch/ideea_gemma/{results_gm,blind_fix,blind_fix_full,blind_fix_keys}` (not in git).

## 1. Task A: the fix and what it changed (label-free)
- Fix: remove non-informative references (whole-string regex, list checked by hand, in the prereg) from both reference sets. TRAIN: 42/1127 correct refs removed, 0/1288 incorrect, 39 questions affected; VAL 5/285.
- The fix changes the objective by very little: cos(grad r_fix, grad r_old) = 0.995 (summed log-prob), 0.993 (length-normalised); cos(ngref-fix lam 1, old ngref) = 0.997.
- **The original diagnosis was wrong.** Old ngref changes log p("I have no comment") by only -0.03 per unit G-norm. That gradient is already saturated: the base model prefers this answer for 71% of questions. What the direction actually does is raise "No, ..." correct refs (+5.2) and push incorrect refs down (-9.6). The extra abstentions are therefore indirect: suppressing assertive answers leaves the prompt's default "I have no comment". Removing those references could not fix this.
- Variants: summed vs length-normalised log-probs; ridge lam_rel in {0.1, 1, 10}; layers {9, 13, 17}; seeds 0/1/2 for the unlabeled Fisher samples.

## 2. Tuning budget and dev selection (pre-registered rule: dev KL <= 1.2, dev refusal <= base 0.722, maximise official dev TI)
- Number of dev configs per family: ngref-fix 72, eucref-fix 48, IDEEA 59, ITI 14 (IDEEA + ITI = 73, including the earlier strength sweeps).
- Strength for ngref-fix/eucref-fix was set label-free to target dev KLs (KL scan on 40 dev questions).
- Picks:
  - ngref-fix: L13, summed, lam_rel 10 (the most Euclidean option in the grid), c = 0.385 (dev TI 0.395, KL 0.45, refusal 0.61). Only 18/72 configs pass the refusal cap. The summed variant with lam <= 1 still raises refusal to 0.79-0.88.
  - eucref-fix: L13, summed, c = 0.329.
  - IDEEA and ITI picks are unchanged under the new rule (IDEEA min_perp K24 a4 nc2 = paper config; ITI K16 a6).
  - VAL-only sensitivity check: L17 nfsum10 (dev TI 0.33).
- Matched strength = dev KL 0.518 (IDEEA's dev KL). On test it gives KL 0.46 for ngref-fix and 0.52 for eucref-fix.

## 3. Results (test split). Official = 395 questions, the two official 7B judges. Strict = 150 random test questions, blind Sonnet labels, mean of 2 passes, after the truncation correction in Section 4
| method (test KL) | official T / I / TI | official dTI vs base | refusal | words | double-neg | MC1 (d) | **strict T / I / TI [95% CI]** | strict - IDEEA | strict - ITI |
|---|---|---|---|---|---|---|---|---|---|
| base (0) | .878/.397/.276 | | .71 | 5.7 | .00 | .385 | .873/.320/**.193** [.133,.260] | -.107 | +.023 |
| IDEEA paper = dev cfg (.514) | .944/.628/.572 | +.296±.025 | .31 | 18.9 | .01 | .435 (+.05) | .870/.390/**.300** [.230,.373] | | +.130 [.070,.193] |
| ITI dev-sel K16 a6 (.662) | .916/.547/.463 | +.187±.024 | .63 | 9.8 | .00 | .377 (-.01) | .860/.257/**.170** [.117,.230] | -.130 | |
| CAA paper L12 a1 (.013, native) | .906/.395/.301 | +.025±.009 | .73 | 6.2 | .00 | | .890/.303/**.197** | | |
| random dirs (3 seeds, KL<=.1; earlier runs) | | -.04..-.06 at equal KL | | | | -.01 | not labelled | | |
| ngref-old (.436) | .957/.246/.203 | -.073±.018 | .85 | 5.6 | .04 | .501 (+.12) | .880/.110/**.060** | -.240 | -.110 |
| **ngref-fix matched (.462)** | .959/.456/.415 | +.139±.022 | .61 | 10.3 | .20 | .529 (+.14) | .767/.217/**.150** [.093,.207] | **-.150 [-.227,-.077]** | -.020 [-.077,.037] |
| ngref-fix native c=.385 (.393) | .957/.446/.403 | +.127±.022 | .62 | 9.7 | .20 | .524 (+.14) | (KL within 30% of matched: not labelled, as pre-registered) | | |
| ngref-fix seed 1 matched (.342) | .949/.471/.420 | +.144±.022 | .59 | 9.4 | .16 | .534 | .790/.243/**.160** | -.140 [-.213,-.070] | -.010 |
| ngref-fix seed 1 native target (.436) | .954/.499/.453 | +.177±.024 | .55 | 10.1 | .23 | .529 | .787/.243/.163 | -.137 | -.007 |
| ngref-fix seed 2 matched (.384) | .919/.572/.491 | +.215±.025 | .49 | 9.9 | .18 | .534 | not labelled | | |
| eucref-fix matched (.523) | .965/.881/.848 | +.572±.026 | .12 | 22.2 | .15 | .511 (+.13) | .827/.477/**.413** [.337,.487] | +.113 [.037,.187] | +.243 |
| ng, judge-SUPERVISED variant (.507) | .856/.959/.815 | +.539±.027 | .01 | 17.5 | .05 | .451 (+.07) | .760/.710/**.510** [.437,.587] | +.210 [.133,.283] | +.340 |
| POST-HOC abstention arm matched (.486) | .949/.651/.600 | +.324±.025 | .36 | 13.5 | .32 | .527 (+.14) | .697/.327/**.237** [.170,.303] | -.063 [-.137,.000] | +.067 |

Official TI at equal realised KL 0.514 (interpolated), minus IDEEA paper config: ngref-fix -0.161 [-0.215, -0.109]; ngref-old -0.374; eucref-fix +0.272; supervised ng +0.244; IDEEA CLI-default strength sweep -0.200; ITI-default sweep -0.216. KL 0.1 / 0.3 curves are in the JSON (`report.matched`).
MC1/MC2 (ngref's own objective family): every ngref/eucref variant gives MC1 +0.12..+0.15 (IDEEA +0.05, ITI ~0). MC1 does not separate the fix from the old objective (+0.14 vs +0.12).

## 4. Strict-label protocol, and a correction made after review
- Protocol: 150 test questions (seed 0) x 10 runs. Items were deduplicated by (question, answer), shuffled across runs, and given random ids; the key is stored outside the labelling directory. Each pass was 7-8 Sonnet subagents with the TruthfulQA definitions (prompt archived in `ideea_harness_gemma/blind/labeller_prompt_fix.txt`). The two passes used different shuffles and chunking.
- Agreement between passes, on unique items: kappa truthful 0.91, informative 0.86, TI 0.86. **This is a same-model, same-prompt test-retest figure, not agreement between independent labellers.**
- **Correction made after the adversarial review:**
  - The first labelling set showed answers cut at the first blank line (the `clean()` helper). The official judges saw the full answer. The cut hit IDEEA (120/150 answers) and ITI (108) far more than ngref-fix (4) or base (2).
  - All cut items were relabelled blind with the full answer, in 2 passes, together with 60 uncut control items. On the controls, old and new TI agree 0.97 and the TI rate is identical (0.383), so there is no labeller drift.
  - Effect: IDEEA strict TI rises from 0.223 (cut) to 0.300 (full) and ITI from 0.150 to 0.170. Every other run moves by at most 0.02. **The table above uses the full-answer labels.** The verdict is the same either way: on cut answers, ngref-fix - IDEEA = -0.073 [-0.140, 0.000] and ngref-fix - base = -0.043.
- Seed-1 mislabel: the first set labelled the seed-1 run calibrated to the native target (c 0.3901). The pre-registered matched run (c 0.3674) was added in the supplementary set. Both are reported; they give the same result.

## 5. Verdict against the pre-registered criteria
| criterion | result |
|---|---|
| 1. strict TI(ngref-fix) - IDEEA > 0, CI excludes 0 | **no**: -0.150 [-0.227, -0.077], significantly worse |
| 2. strict TI(ngref-fix) - ITI > 0 | **no**: -0.020 [-0.077, 0.037] |
| 3. refusal(ngref-fix) <= base 0.711 | yes (0.61) |
| 4. both passes give a positive point difference | no (pass 1 / pass 2 vs IDEEA: -0.160 / -0.140) |
| partial success: strict - base > 0, CI excludes 0 | **no**: -0.043 [-0.110, 0.027] |
| robustness: seed-1 has the same sign | yes, same (negative) sign: -0.140 vs IDEEA |
| robustness: official-minus-strict gap not > 2x IDEEA's | gap 0.283 vs IDEEA 0.267 on the same questions: not flagged |
| robustness: no anomalous degeneration | **flagged**: 20% double-negation answers (IDEEA 1%), strict truthfulness falls 0.87 -> 0.77 |

**=> FAILURE.**
- The fix moves ngref off the hedging failure mode on the official judges: official TI goes from 0.20 (old) to 0.41 at equal KL, and refusal falls from 0.85 to 0.61.
- But under strict labels the gain disappears. The answers become tangled negations ("There is no scientific evidence that aspartame is not generally considered safe", occasionally "not not not ..." loops), which strict labellers mark false or uninformative. Strict TI is 0.150, against 0.193 for base and 0.300 for IDEEA.

## 6. What is solid / what is not
Solid:
1. The non-informative-reference fix does not repair judge-free ngref. Strict TI is not above base; it is significantly below IDEEA (-0.15) and level with ITI. This holds for both labelling passes, for seed 1, and with or without the truncation correction.
2. The diagnosis behind the fix was wrong. The abstention references contribute almost nothing directly (saturated gradient). The objective rewards lowering the likelihood of assertive/incorrect-style answers and raising "No/not" phrasing, and at generation time this turns into either "I have no comment" or negation salad.
3. The official judges inflate every steered method (official - strict, same 150 questions: base +0.11, IDEEA +0.27, ITI +0.27, ngref-fix +0.28, ng +0.31, eucref-fix +0.45). IDEEA's paper-level official TI of 0.572 corresponds to a strict 0.300, a real gain of +0.107 [0.040, 0.173] over base.
4. The judge-SUPERVISED ng remains the best method under strict labels: 0.510, +0.21 over IDEEA. It is trained on the official judges, so it is a supervised variant and not in the same supervision class as the others.

Notable, but not established:
- **eucref-fix** (plain gradient of the same reference objective, judge-free, same supervision class as IDEEA): strict 0.413, +0.113 [0.037, 0.187] over IDEEA at matched KL; it was a pre-registered comparison method. However:
  - it is essentially the old eucref (cos 0.995), so the fix is not the cause; the earlier strict 0.20 was measured at KL 1.09, where outputs are degenerate;
  - its official - strict gap is the largest (0.45), and it has 15% double negations;
  - its strict truthfulness drops (0.87 -> 0.83) while informativeness rises (0.32 -> 0.48);
  - n = 150, a single seed, and one labeller model.
  So "the Euclidean reference gradient at moderate KL beats IDEEA" is a hypothesis to re-test with fresh questions, a second labeller family and seed replication, not a claim.
- **Post-hoc abstention arm** (not pre-registered; natural gradient of r_fix - 1 x log p("I have no comment."), same dev rule over 12 configs):
  - official TI 0.600 at KL 0.49 (IDEEA 0.572), but strict 0.237: +0.04 vs base (n.s.), -0.06 vs IDEEA;
  - 32% double negations; strict truthfulness 0.70;
  - with a penalty of 10 the refusal rate goes *up* (dev 0.73-0.89), so first-order control of the abstention fails.
  Penalising abstention alone does not make ngref informative; it redirects it into negation salad.
- Seed variance of ngref-fix is large on the official judges (matched TI 0.415 / 0.420 / 0.491; seed-2 refusal 0.42-0.49 vs 0.59-0.61). Seed 2 was not strict-labelled.

Adversarial self-review (Opus reviewer + my own checks):
- No leakage found. The fit uses only TRAIN references plus unlabeled samples; selection uses dev only; test was never used for a choice. One exception: the strengths for ng and old ngref (c 0.54, 0.74) were interpolated from their *test* KL curves. This is label-free and was declared in the prereg.
- The selection-rule code matches the prereg.
- Bootstrap is over questions and paired; the CI code was checked.
- Found and fixed: the truncation bias against IDEEA/ITI in the strict set (Section 4) and the seed-1 run mislabel.
- Remaining caveats:
  - labeller "independence" is test-retest of one model;
  - the dev picks for ngref-fix and eucref-fix sit at L13, so the layer grid did not matter;
  - IDEEA was tuned over 59 configs and ngref-fix over 72 (comparable; ngref-fix is not disadvantaged);
  - power: paired strict SE ~0.035, so differences under ~0.07 cannot be resolved.

## 7. Next (not run)
The reference-likelihood objective rewards "sounding like a correct reference" (negation-heavy) rather than "answering". A judge-free repair would have to reward answer content, for example:
- contrast the correct vs incorrect references conditional on the same answer prefix or entity;
- or score log p(correct) - log p(incorrect) only on content tokens, with negation-flip pairs as hard negatives;
- and add a fluency/degeneracy constraint, e.g. a KL-to-base cap on the negation-token distribution.
Following the pivot-not-patch rule, any such repair should be pre-registered and tested on new questions, not patched into this objective. eucref at moderate KL deserves a pre-registered replication (fresh questions, a non-Claude second labeller, 3 seeds).

## Compute
GPU-hours (A100 3g.20gb), jobs 6410963/6410964 (refs + samples), 6415772/6415773 (dev grid), 6418711 (abstention features), 6423464 (seed-2 samples), 6427371 (test), 6427374 (post-hoc dev), 6427391 (seeds), 6578224 (post-hoc test): **8.0 GPU-h**. At most 2 jobs ran concurrently. The 7B judges were used for evaluation only, one at a time.
Sonnet subagents for labelling: 15 + 10 agents. One Opus reviewer.

# Pre-registration: ngref objective fix + fair comparison (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

Written 2026-10-19, **before any run** of the fixed objective (dev or test). Code: `ideea_harness_gemma/scripts/gm_fix.py` (+ `fx` spec in `gm_eval.py`). Everything else (split, prompt, greedy 50 tokens, bf16, official judges one at a time, realised per-token KL definition, hook protocol) is exactly as in `results/unified_gemma2b.md`.

## 1. The fix (Task A)
Diagnosis (established before this run): judge-free ngref optimises r = mean log p(correct refs) - mean log p(incorrect refs). The TruthfulQA *correct* references contain content-free abstentions ("I have no comment", 86/790 questions), the incorrect references essentially never do; the shared abstention text adds coherently across questions, so the fitted direction raises the "I have no comment" rate (Gemma greedy 71% -> 79%).

Fix: remove **non-informative references** from BOTH the correct and the incorrect set before fitting. Rule = whole-string regex (case-insensitive, optional final period):
`I have no comment | No comment | I don't know (for sure) | I do not know | Unknown | It is uncertain | It is unknown | It is not known (with certainty) | I am an AI and I don't know the answer | I have no personal knowledge | Nobody knows | No one knows`.
Manual check of all short references containing comment/know/unknown/uncertain/depends (done on the CSV, before fitting): matched set = {"I have no comment", "I  have no comment", "I don't know", "I don't know for sure", "I am an AI and I don't know the answer", "I have no personal knowledge", "It is not known with certainty", "It is uncertain", "Unknown" (correct and incorrect)}. **Kept on purpose** (question-specific content, not a generic abstention): "It depends on which country you are in", "It depends on your current location", "I don't know what country you're in", "The origin of the phrase is not known", "We don't know if intelligent life exists elsewhere", etc.
Counts: TRAIN (316 q) correct refs 1127, removed 42; incorrect 1288, removed 0; 39 train questions affected; no train question loses all its correct refs. VAL (79 q): 5 of 285 correct removed. (TEST, for information only, never used in fitting: 59/1365 correct, 2/1630 incorrect.) MC1/MC2 evaluation keeps the standard full reference set (benchmark definition unchanged).

Two objective variants: `sum` (summed token log-prob, as before) and `len` (log-prob divided by the reference's token count). Directions: ngref-fix = (G + lam I)^-1 grad r_fix, eucref-fix = grad r_fix, G = within-question Fisher of 16 unlabeled temperature-1 base samples per TRAIN question at the same layer (no labels). Supervision: TRAIN-split reference answers only (same data class as IDEEA/ITI/CAA); no judge, no judged generations.

## 2. Dev-only tuning grid (same tuning budget class as IDEEA/ITI)
- layer L in {9, 13, 17} (Gemma-2-2B has 26 layers; 13 was the previous fixed choice)
- objective variant in {sum, len}
- ridge lam_rel in {0.1, 1, 10} (lam = lam_rel * tr(G)/d; ngref only)
- strength: instead of a raw c grid (the KL(c) curve of ngref is very steep and layer dependent), c is calibrated label-free to target realised dev KL in {0.1, 0.3, 0.6, 1.0} for ngref-fix and {0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0} for eucref-fix, by a KL scan (c in {0.1,0.2,0.4,0.8,1.6}) on the first 40 development questions (generation + KL only, no judge), log-log interpolation.
- config counts: ngref-fix 3 x 2 x 3 x 4 = **72** dev configs; eucref-fix 3 x 2 x 8 = **48**; (IDEEA 54 + ITI 9 = 63 dev configs already run, plus the 5-strength CLI-default sweeps.)
- All dev configs are run on the full development split (395 q = train 316 + val 79; this is in-sample for the fitted directions, exactly as IDEEA/ITI's directions are fitted and tuned on the same development split) and judged with the official judges.

## 3. Selection rule on dev (applied identically to every family, including re-applying it to the existing IDEEA/ITI dev grid)
Among a family's dev configs with realised dev KL <= 1.2 and dev refusal rate <= dev base refusal rate (0.722; refusal = `is_refusal` regex as before), pick the max official dev TI; ties (within 0.005) -> lower KL. Report whether IDEEA/ITI's selected configs change under this rule (previous rule had no constraints; IDEEA's previous pick, min_perp K24 a4 nc2, has dev KL 0.518, refusal 0.311).
Sensitivity (reported, not used for the claims): selection on the 79 VAL questions only (held out from the fit).

## 4. Test runs (each once, after selection; no further changes)
For the selected (layer, variant, lam) of ngref-fix and of eucref-fix: (a) the 4 (resp. 8) calibrated strengths -> KL curve; (b) the dev-selected strength ("native"); (c) **matched**: c giving dev KL = 0.518 (IDEEA paper/dev-selected config's dev KL; its test KL is 0.514), interpolated from the dev KL sweep of that direction. Seeds: the selected ngref-fix config refitted with unlabeled-sample seeds 1 and 2 (new G), strength re-calibrated on dev to KL 0.518 and to the native target, run on test. Supervised ng and old ngref at KL ~0.51: c interpolated from their existing test KL curves (label-free), ng c=0.54, old ngref c=0.74. MC1/MC2 for ngref-fix/eucref-fix native + matched + seeds.

## 5. Metrics
(1) official truth / info / TI (395 test q, paired bootstrap 2000); equal-KL interpolation as before; (2) refusal rate, mean words, plus degeneracy diagnostics: double-negation rate (>=2 negations "not/no/never/n't" in one sentence with "no/not ... not"), repeated 3-gram rate, hit-the-50-token-limit rate; (3) MC1/MC2 (note: ngref's optimised objective family); (4) **strict independent check**: blind Claude (Sonnet) subagent labels with the TruthfulQA definitions of truthful and informative; 150 random test questions (seed 0) shared by all labelled runs; items from all runs shuffled together and renumbered randomly; run identity hidden; answer key stored outside the labelling directory; two independent labeller passes (different shuffles and different chunking); report per-pass rates, Cohen's kappa between passes (truthful, informative, TI), and item score = mean of the two passes.
Labelled runs: base, IDEEA paper cfg, ITI (rule-selected), CAA paper cfg (L12 a=1, native), ngref-old @KL~0.51, ngref-fix @matched KL, ngref-fix native (if its dev KL differs from matched by >30%), ngref-fix seed-1 @matched, eucref-fix @matched, ng (judge-supervised, labelled SUPERVISED variant) @KL~0.51.

## 6. Success / failure (decided now)
**Primary success** requires ALL of:
1. strict TI(ngref-fix @matched) - strict TI(IDEEA paper cfg) > 0 with paired-bootstrap 95% CI excluding 0 (150 q);
2. same vs ITI (rule-selected);
3. test refusal rate of ngref-fix @matched <= base refusal (0.711);
4. both labeller passes individually give a positive point difference for 1 and 2.
**Partial success**: strict TI(ngref-fix) - base > 0 with CI excluding 0 and refusal <= base, but 1 or 2 fails.
**Failure**: strict TI(ngref-fix) - base CI includes 0 or is negative, or refusal > base.
Robustness flags (reported, downgrade the claim if violated): seed-1 refit gives the same sign vs IDEEA; official-minus-strict TI gap of ngref-fix not > 2x IDEEA's gap (judge exploitation); mean length / double negation / repetition not anomalous relative to IDEEA.
Power note: with 150 paired questions the SE of a strict TI difference is ~0.05, so only differences of ~0.1 or more can be established.

Honest-risk note written in advance: removing 42 of 1127 train correct references may not remove the hedging pressure (other hedge-like correct references remain, and nothing in r rewards informativeness), so failure is a real possibility.

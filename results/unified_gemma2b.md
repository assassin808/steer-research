# Unified evaluation on gemma-2-2b-it in IDEEA's official TruthfulQA protocol

Date: 2026-10-19. Code: `ideea_harness_gemma/` (copy of IDEEA commit 07ced80 + documented patches; `ideea_harness/` and `~/ext/IDEEA` untouched). Numbers: `results/unified_gemma2b.json`. Raw generations/judgements and per-item MC are in `~/scratch/ideea_gemma/results_gm/` (not in git).

## Protocol (what is identical to IDEEA, what is ours)
- Model google/gemma-2-2b-it (26 layers, 8 heads x 256; the IDEEA head hook handles o_proj 2048 vs hidden 2304 via `ModelDims.head_width`). IDEEA's own split (seed 0): dev 395 / **test 395**; prompt, greedy decoding, 50 new tokens, bf16.
- Judges: the official allenai/truthfulqa-{truth,info}-judge-llama2-7B, IDEEA's prompts and rule (top token "yes" with p>0.5), loaded **one at a time** (bf16) by patching `TruthfulQA.judge`. Batch invariance checked: re-judging 600 answers at batch size 1 flips 0 truth and 1 info labels. TI = truth AND info per item.
- Fidelity: IDEEA's own CLI (`cli.run_eval`, batch 1, defaults alpha=15, k=48) vs our batched pipeline (TI): base 0.273 vs 0.276; ITI-15 0.051 vs 0.063; IDEEA-15 0.094 vs 0.099; CAA L12 a=1 0.296 vs 0.301. Compared with the IDEEA paper's Gemma2-2B numbers (Table 1/7): base T/I/TI .886/.392/.279 (ours .878/.397/.276); ITI .463 (ours, paper config K=24 a=4: .448); CAA .294 (ours .301); IDEEA min-perp .572 (ours with the paper's best config K=24, a=4, nc=2: **.572**). Reproduction is good.
- Realised KL = mean per-token KL(steered||base) along the steered greedy answer (token-averaged within answer, then over questions).
- Our methods act at the residual stream of layer 13, last position of every forward call (IDEEA's CAA hook protocol), step size c x (mean residual norm). **ngref**: v = (G + lam I)^-1 grad_theta r, r = mean log p(correct refs) - mean log p(incorrect refs), G = Fisher from 16 unlabeled base samples x 316 train questions, lam chosen on val by first-order gain (rel. 1 x tr(G)/d); supervision = TruthfulQA reference answers of the 316 TRAIN questions only; no judge. **eucref**: v = grad r (no metric). **ng / euc** (labelled *supervised by the official judges*): same machinery with s = truth AND info of the base samples judged by the official judges. Random: Gaussian directions, 3 seeds at c in {0.05,0.2,0.8} (2 seeds at others). Test questions are disjoint from train/val (checked).
- IDEEA family: (a) **CLI defaults** alpha=15, k=48, nearest_pos_neg nc=4; (b) strength sweep alpha 0.5..15 at defaults; (c) **paper hyper-parameter grid** (K in {16,24}, alpha in {2,4,6}, nc 2-4, three variants; ITI K in {8,16,24}) run on the **development** split, best config per method picked by dev TI, then run once on test; the dev-selected IDEEA best (min_perp, K=24, a=4, nc=2) coincides with the paper's own Table 8 choice. CAA: layers 8/12/13/18, several strengths.
- Statistics: 395 test questions, deterministic greedy decoding, so the unit of noise is the question: SEs are paired bootstrap over questions (2000 resamples). Equal-KL comparisons interpolate per-item outcomes linearly in log KL between the two bracketing strengths of one method (same questions, so paired). "seed" variation: random directions (3), ngref refit with new unlabeled samples (seed 1); IDEEA/ITI directions are deterministic at seed 0 (as the task requires).

## Result 1: official-judge truth x info (TI), Delta vs base (base TI 0.276, truth 0.878, info 0.397, refusal 0.71, 5.7 words)

Equal realised KL (paired bootstrap SE; dense strength sweeps of each method; IDEEA/ITI = CLI-default k=48, nearest_pos_neg nc=4):

| method | KL 0.03 | KL 0.1 | KL 0.3 | KL 1.0 |
|---|---|---|---|---|
| ITI (k48) | +0.012±.009 | +0.015±.013 | +0.055±.015 | +0.111±.023 |
| IDEEA (defaults, sweep) | +0.003±.009 | +0.017±.013 | +0.070±.016 | +0.011±.018 |
| CAA L13 | -0.060±.012 | -0.106±.015 | -0.159±.018 | -0.227±.020 |
| random (3 seeds) | -0.040±.009 | -0.059±.011 | | |
| **ngref (judge-free)** | -0.005±.011 | -0.045±.014 | -0.073±.016 | -0.112±.019 |
| **eucref (judge-free)** | +0.078±.013 | +0.146±.018 | +0.314±.022 | +0.557±.026 |
| ng (judge-supervised) | +0.122±.017 | +0.291±.021 | +0.536±.026 | +0.541±.029 |
| euc (judge-supervised) | +0.194±.017 | +0.261±.023 | +0.485±.026 | |

Native strengths / tuned IDEEA (test, one run each):

| config | KL | truth | info | TI | dTI | refusal | MC1 d | MC2 d |
|---|---|---|---|---|---|---|---|---|
| base | 0 | .878 | .397 | .276 | | .71 | (.385) | (.587) |
| ITI CLI default (a=15,k=48) | 2.68 | .711 | .172 | .063 | -.213±.025 | .00 | -.114 | -.109 |
| IDEEA CLI default (a=15,k=48) | 2.43 | .919 | .159 | .099 | -.177±.025 | .05 | -.035 | -.021 |
| CAA L13 a=15 | 2.29 | .896 | .013 | .003 | -.273±.023 | .00 | -.154 | -.092 |
| CAA L12 a=1 (paper cfg) | 0.013 | .906 | .395 | .301 | +.025±.009 | .73 | +.018 | +.018 |
| ITI K=24 a=4 (paper cfg) | 0.455 | .896 | .552 | .448 | +.172±.022 | .59 | +.018 | +.040 |
| ITI K=16 a=6 (dev-selected) | 0.662 | .916 | .547 | .463 | +.187±.024 | .63 | -.008 | +.013 |
| IDEEA min-perp K=24 a=4 nc=2 (paper cfg = dev-selected) | 0.514 | .944 | .628 | **.572** | +.296±.025 | .31 | +.051 | +.052 |
| IDEEA nearest-cluster K=24 a=4 nc=2 (dev-sel.) | 0.522 | .924 | .557 | .481 | +.205±.022 | .37 | +.025 | +.030 |
| IDEEA nearest_pos_neg K=16 a=6 nc=2 (dev-sel.) | 0.453 | .911 | .592 | .504 | +.228±.024 | .74 | +.010 | +.025 |
| eucref c=0.2 | 0.256 | .952 | .600 | .552 | +.276±.024 | .52 | +.142 | +.129 |
| eucref c=0.4 | 1.088 | .992 | .853 | .846 | +.570±.028 | .02 | +.127 | +.127 |
| ngref c=0.2 | 0.016 | .916 | .375 | .291 | +.015±.011 | .73 | +.134 | +.116 |
| ngref c=0.4 | 0.060 | .929 | .319 | .248 | -.028±.014 | .79 | +.162 | +.143 |
| ng (sup.) c=0.2 | 0.144 | .830 | .800 | .630 | +.354±.025 | .31 | +.053 | +.049 |
| ng (sup.) c=0.4 | 0.327 | .856 | .967 | .823 | +.547±.028 | .05 | +.061 | +.071 |

Tuned IDEEA vs ours at the SAME KL (family minus IDEEA config, TI / MC1): at KL 0.51 (IDEEA min-perp, paper cfg) eucref +0.147±.025 / +0.083±.018, ng +0.254±.028, ngref -0.379±.026 / +0.070±.019, default-IDEEA sweep -0.200±.026. So under the official judges the tuned IDEEA is the best of the judge-free-trained published baselines (+0.30), better than ITI (+0.17..0.19), and is beaten at equal KL by eucref and the judge-supervised ng, but NOT by judge-free ngref.

## Result 2: judge-free multiple choice (MC1/MC2 with IDEEA's prompt, hooks acting as in generation, 395 test q). Base MC1 0.385, MC2 0.587.

Equal-KL Delta MC1 (MC2 in brackets mostly the same sign/size):

| method | KL 0.03 | KL 0.1 | KL 0.3 | KL 1.0 |
|---|---|---|---|---|
| ITI (k48) | +.006±.011 | -.001±.013 | -.025±.015 | -.057±.020 |
| IDEEA (defaults) | +.025±.010 | +.044±.015 | +.051±.015 | +.025±.018 |
| CAA L13 | -.026±.010 | -.047±.014 | -.077±.016 | -.123±.020 |
| random (3 seeds at c=.2) | -.014±.009 | -.019±.012 | | |
| **ngref** | **+.147±.019** | **+.152±.020** | +.131±.021 | |
| eucref | +.093±.014 | +.126±.019 | +.140±.020 | +.127±.023 |
| ng (supervised) | +.025±.009 | +.046±.012 | +.060±.015 | |

ngref - IDEEA(default sweep) at equal KL: MC1 +0.122±.017 (KL 0.03), +0.109±.019 (0.1), +0.080±.019 (0.3); ngref - ITI +0.141..0.156. ngref - eucref: +0.054±.011 (0.03), +0.026±.011 (0.1), -0.009±.013 (0.3): natural gradient helps only at small KL. Seed check (ngref refit with new unlabeled samples): MC1 0.519/0.509 (c=0.2), 0.547/0.549 (c=0.4): stable; TI 0.291/0.276, 0.248/0.253 (paired diff 0.015±.009, -0.005±.010); at c=0.8 the realised KL differs 3x between fits (0.83 vs 0.28) so strong steps are not reproducible.

## What is solid, what is not
Solid (large vs SE, replicated by the Qwen3-1.7B study in `track1_judgefree.md`):
1. **Judge-free ngref improves held-out MC1/MC2 by +0.13..+0.16 at per-token KL 0.016-0.06, 3-5x more than IDEEA/ITI at any strength and 10x more than random (which is ~-0.01 +- 0.01).** This is a likelihood readout of the quantity ngref optimises (log p of correct vs incorrect references), on disjoint questions.
2. **ngref does NOT improve generation TI** with the official judges (dTI between -0.03 and +0.015 up to KL 0.06, negative beyond); truth rises (0.878 -> 0.93) while info falls (0.40 -> 0.32) and the refusal rate rises (0.71 -> 0.79): truth is gained by answering "I have no comment" more often. Same conclusion as on Qwen.
3. Paper-config IDEEA reproduces (TI 0.572 vs paper 0.572), IDEEA defaults (a=15) are far too strong for this model (TI 0.099 < base), CAA is at best +0.03 and otherwise negative, random directions never help.
4. Judge-supervised ng/euc (and the judge-free eucref) raise the official TI far more than IDEEA at equal KL (see Result 1) -- but see the audit.

Not solid / caveats (adversarial self-review):
- **Official-judge TI is gameable and inflated for exactly the methods that win it.** Blind strict labels (a Claude Sonnet subagent, TruthfulQA definitions, answers shuffled across runs, n=60 questions x 10 runs, SE ~0.06; one labeller, no inter-rater check): strict TI vs official TI: base .22 vs .38, IDEEA min-perp .38 vs .60, ITI K24 .27 vs .58, **eucref c=0.4 .20 vs .75**, eucref c=0.2 .30 vs .53, ngref c=0.4 .13 vs .30, ng c=0.2 .43 vs .65, ng c=0.4 .55 vs .78. eucref c=0.4's outputs are double-negative word salad ("There is no scientific evidence that ghosts have not been proposed ... they are not a product of the imagination") that the official judges mark truthful AND helpful (strict agreement with the official info label on that run: 47%). So eucref's +0.57 is mostly judge exploitation, not a real gain; the judged ng (strict 0.43/0.55 vs base 0.22) is the only method whose gain survives the strict audit by a clear margin, and it is trained on those very judges (supervised), so this is the expected direction; IDEEA min-perp's gain survives only partially (+0.16 strict vs +0.22 official on the same 60 questions; +0.30 on all 395).
- Because the official info judge accepts hedged text of 15-30 words, longer answers (nwords 5.7 -> 9..28) systematically score higher; "refusal suppression" explains part of every TI gain (refusal 0.71 -> 0.31 for IDEEA min-perp, 0.05 for ng c=0.4). Truth-by-brevity: base truth .878 is itself high because 71% of answers are "I have no comment", and ngref/CAA raise truth by raising that rate further.
- MC is the optimised objective for ngref/eucref (different questions, same reference style). It is a within-objective generalisation check, not evidence of more truthful generations. IDEEA/ITI/CAA are fitted on a different (probe/mass-mean) objective; MC is therefore not a neutral yardstick between families. MC is also scored with the hook on every position of the answer, not only on the generation path, so it is a stricter test than the generation KL suggests, and realised KL (measured on greedy answers, 71% of which are "I have no comment") is a weak measure of the size of the intervention: the same step moves MC far more than it moves the tokens of those short answers.
- Tuning asymmetry: IDEEA/ITI were tuned over 63 configs on the dev split (in-sample for their directions, which were fitted on the same dev questions; the paper's choice coincides with ours); ngref/eucref/ng strengths were swept but never selected on test. Our layer (13) and ridge (rel. 1) were fixed once, not tuned for TI.
- Single fit seed pair for ngref; IDEEA/ITI/CAA deterministic; differences below ~0.02 TI / 0.03 MC are within noise. Strict-label audit has n=60 per run and one labeller.
- Not run: IDEEA with auto-nc / nc 5-6 and K=8 for IDEEA (only ITI K=8); 5-fold cross-fit as in the paper (we select on dev, evaluate on test once); a second model; layer sweep for our methods; ng judged fit used the official judges on 316 train + 79 val questions x 16 samples.

## Compute
GPU-hours (A100 3g.20gb slices): this session 8.0 h (jobs g5-g11: MC 1.9+0.8, seed/ngref refit 0.9, IDEEA grid + judging 2.9, CLI fidelity 0.7, judge check 0.2, dev-selected runs 0.4); earlier agent that built the pipeline (fit, test grid, judging) 2.9 h (jobs g1-g4, g4 ended in the MC bug that is fixed). Total ~10.9 GPU-hours. Plus one Sonnet subagent for blind labelling (600 items).

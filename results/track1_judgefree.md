# Track 1b — judge-free natural-gradient steering ("ngref") in IDEEA's TruthfulQA harness (Qwen3-1.7B)

Date: 2026-10-19 (jobs ran 2026-10-15, Slurm 5735425 / 5735509). Same harness, split (seed 0, 395 test questions), layer 14, proxy judges and realized-KL definition as `track1_ideea_harness.md`.

## Method
Direction v = (G + lam I)^-1 grad_theta r, with r(x) = mean log p(correct references | x) - mean log p(incorrect references | x) (teacher-forced), G = sequence Fisher of the same intervention from unlabeled base-model samples, lam = 3 x trace(G)/d. Supervision = reference answers of the 316 TRAIN questions only (same as IDEEA/ITI/CAA); **no judge, no judged generations**. Variants: `eucref` (v = grad r, no metric); `...E` = fitted with the embedding-scored reference variant. Files: `ideea_harness/scripts/ngref_fit.py`, `results_t1/ref/*.json`, `results_t1/mc_summary.json`, `logs/jobJ_5735425.log`.

## Results (realized per-token KL; base: truth_qwen 0.608, info 0.681, TI_qwen 0.289, MC1 0.304, MC2 0.491)

| run | KL | truth (qwen) | info | TI qwen | MC1 (Δ) | MC2 (Δ) |
|---|---|---|---|---|---|---|
| ngref c=0.05 | 0.011 | 0.618 | 0.643 | 0.261 | 0.342 (+0.04) | 0.527 (+0.04) |
| ngref c=0.1 | 0.039 | 0.646 | 0.615 | 0.261 | 0.365 (+0.06) | 0.546 (+0.06) |
| ngref c=0.2 | 0.167 | 0.727 | 0.527 | 0.253 | 0.403 (+0.10) | 0.596 (+0.11) |
| ngref c=0.4 | 0.865 | 0.734 | 0.441 | 0.175 | 0.415 (+0.11) | 0.650 (+0.16) |
| eucref c=0.05 | 0.056 | 0.666 | 0.592 | 0.258 | 0.349 | 0.545 |
| eucref c=0.1 | 0.195 | 0.706 | 0.514 | 0.220 | 0.377 | 0.574 |
| eucref c=0.2 | 1.079 | 0.810 | 0.349 | 0.159 | 0.423 | 0.627 |
| ngrefE c=0.2 | 0.148 | 0.623 | 0.595 | 0.218 | 0.372 | 0.561 |
| ngrefE c=0.4 | 0.385 | 0.615 | 0.582 | 0.197 | 0.400 | 0.613 |

Existing baselines, MC Δ vs base (SE 0.009-0.015): ITI α=1 (KL 0.143) +0.025 / +0.035; ITI α=2.5 (KL 1.05) +0.048 / +0.062; IDEEA α=1 (KL 0.07) +0.013 / +0.017; IDEEA α=2.5 (KL 0.57) +0.025 / +0.036; CAA −0.013…−0.035 (MC1); random −0.008…−0.023 (MC1); judged ng/ngkl 0.000 ± 0.01.

## What it shows
1. **Likelihood readout: strong.** At KL ≈ 0.04-0.17, ngref raises MC1 by +0.06…+0.10 and MC2 by +0.06…+0.11, versus +0.025 / +0.035 for ITI at KL 0.14 and ≤ +0.05 / +0.06 for ITI at KL 1.05 or IDEEA at any strength. Natural gradient beats the plain gradient at equal KL (ngref MC1 0.403 at KL 0.17 vs eucref 0.377 at KL 0.20; eucref needs KL 1.08 for 0.423).
2. **Generation: truth up, informativeness down, TI not up.** truth_qwen 0.608 → 0.73 at KL 0.17, but info 0.68 → 0.53 and TI_qwen 0.289 → 0.253. The judge-free direction makes the model hedge/shorten (truth gained by saying less); the judged `ng` of Track 1 (TI +0.21 at KL 0.05) gets its gain from informativeness, which the reference-likelihood objective does not reward. So judge-free ngref does **not** reproduce the generation win of judged ng.
3. Beyond c = 0.4 the model degrades (c = 0.8: truth 0.40, KL 2.7).

## Caveats (important)
- MC1/MC2 is the same quantity ngref optimizes (log p correct vs incorrect references), fitted on train questions and evaluated on held-out test questions; it is a within-objective generalization check, not independent evidence of "more truthful behaviour". The independent generation check (proxy judges) shows no TI gain.
- Proxy judges only (Qwen3-1.7B / embedding); no Claude blind labels were collected for ngref runs; no ngref-vs-baseline paired SEs computed yet for generation; single seed; one layer.
- IDEEA's own hyperparameters were not re-tuned: job K only ran the self-check (16/16 identical responses, KL reproduces exactly) and the 5-fold prep (mean top-48 head overlap across folds 0.72). The IDEEA strength sweep was not run.
- A fit/eval harness difference to remember: ngref `pred_kl_globalG` overpredicts realized KL by 4-10x at large c; all tables above use realized KL.

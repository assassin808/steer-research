# ngref objective fix + fair comparison (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

**Status: PARTIAL (2026-10-20).** Official-judge test numbers, dev selection and the post-hoc abstention arm are done; the blind strict labels (2 passes) — which the pre-registered success criterion depends on — are still being collected. No verdict yet.
Pre-registration: `results/ngref_fix_prereg.md` (committed 2f6224e before any dev/test run; unchanged). Code: `ideea_harness_gemma/scripts/gm_fix*.py`, `gm_blind_build.py`. Raw: `~/scratch/ideea_gemma/results_gm/{development_fix,test_fix,test_fix_s1,test_fix_s2,mc*}`, `fix_selection.json`, `fix_report.json`.

## Fix (Task A) and label-free diagnostics
- Removed non-informative references (whole-string regex, manually checked list in the prereg): TRAIN 42/1127 correct, 0/1288 incorrect, 39 questions affected; VAL 5/285 correct.
- The fix barely changes the direction: cos(g_filtered, g_old) 0.995 (sum), 0.993 (len); cos(ngref-fix sum lam1, old ngref) 0.997.
- Mechanism (first-order dlog p per unit G-norm, train refs): old ngref moves log p("I have no comment") by only -0.03 (already saturated), raises "No, ..." correct refs (+5.2) and pushes incorrect refs down (-9.6). So the abstention refs were not the direct driver; the hedging is indirect (suppressing assertive answers leaves the prompt's default abstention). The prior diagnosis was only partly right.

## Dev selection (pre-registered rule: dev KL<=1.2, dev refusal<=0.722, max official dev TI)
| family | dev configs | passing constraints | selected | dev TI | dev KL | dev refusal |
|---|---|---|---|---|---|---|
| ngref-fix | 72 | 18 | L13, sum, lam_rel 10, c=0.385 | 0.395 | 0.45 | 0.61 |
| eucref-fix | 48 | 37 | L13, sum, c=0.329 | 0.911 | 0.77 | 0.01 |
| IDEEA | 59 | 28 | min_perp K24 a4 nc2 (= paper cfg, unchanged) | 0.527 | 0.52 | 0.31 |
| ITI | 14 | 11 | K16 a6 (unchanged) | 0.446 | 0.67 | 0.63 |
| post-hoc abstention arm | 12 | 7 | L13 sum lam10, b=1, c=0.378 | 0.547 | 0.61 | 0.34 |
Configs with sum variant and lam<=1 raise the refusal rate (0.79-0.88, like old ngref) and fail the cap; the picked lam=10 is the most Euclidean option. VAL-only selection would pick L17 nfsum10.

## Test, official judges (395 q; paired bootstrap SE)
| run | KL | truth | info | TI | dTI vs base | refusal | words | double-neg | MC1 (d) |
|---|---|---|---|---|---|---|---|---|---|
| base | 0 | .878 | .397 | .276 | | .71 | 5.7 | .00 | .385 |
| IDEEA paper/dev cfg | .514 | .944 | .628 | .572 | +.296±.025 | .31 | 18.9 | .01 | .435 (+.05) |
| ITI dev-sel K16 a6 | .662 | .916 | .547 | .463 | +.187±.024 | .63 | 9.8 | .00 | .377 (-.01) |
| CAA paper L12 a1 | .013 | .906 | .395 | .301 | +.025±.009 | .73 | 6.2 | .00 | |
| ngref-old c=.74 | .436 | .957 | .246 | .203 | -.073±.018 | .85 | 5.6 | .04 | .501 (+.12) |
| **ngref-fix matched** c=.401 | .462 | .959 | .456 | .415 | +.139±.022 | .61 | 10.3 | .20 | .529 (+.14) |
| ngref-fix native c=.385 | .393 | .957 | .446 | .403 | +.127±.022 | .62 | 9.7 | .20 | .524 (+.14) |
| ngref-fix seed1 matched | .436 | .954 | .499 | .453 | +.177±.024 | .55 | 10.1 | .23 | .529 |
| ngref-fix seed2 matched | .484 | .919 | .630 | .549 | +.273±.026 | .42 | 10.8 | .25 | .532 |
| eucref-fix matched c=.267 | .523 | .965 | .881 | .848 | +.572±.026 | .12 | 22.2 | .15 | .511 (+.13) |
| ng (judge-SUPERVISED) c=.54 | .507 | .856 | .959 | .815 | +.539±.027 | .01 | 17.5 | .05 | .451 (+.07) |
| post-hoc abst. arm matched c=.360 | .486 | .949 | .651 | .600 | +.324±.025 | .36 | 13.5 | .32 | .527 (+.14) |
Equal-KL (0.514) official TI minus IDEEA paper cfg: ngref-fix -0.161 [-0.215, -0.109]; eucref-fix +0.272 [0.223, 0.321]; ng supervised +0.244; ngref-old -0.374.
Seed spread of ngref-fix at matched KL: official TI 0.415 / 0.453 / 0.549 — large (G seeds change the direction's behaviour).
Degeneracy: ngref-fix and the abstention arm produce many double-negation outputs (20-32% vs IDEEA 1%), e.g. "There is no scientific evidence that aspartame is not generally considered safe", occasionally "not not not ..." loops; official-judge TI for these is suspect -> the strict check is decisive.

## Pending
Blind strict labels (2 passes) -> pre-registered verdict; adversarial self-review.

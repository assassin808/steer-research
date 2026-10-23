# IDEEA <-> ngref bridge (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

Date 2026-10-23. Pre-registration: `results/bridge_prereg.md` (commits 13312ad + amendment de288ba, written before any run).
Numbers: `results/bridge.json`. Code: `ideea_harness_gemma/scripts/bridge.py` (features, fit, hooks, arm O), `bridge_select.py`,
`bridge_testspecs.py`, `bridge_report.py`, `bridge_blind_build.py`, `bridge_strict_merge.py`, `br_*.sbatch`.
Raw generations, labels, blind key: `~/scratch/ideea_gemma/{runs_ng/bridge, results_gm/*bridge*, blind_bridge, blind_bridge_keys}` (not in git).
Status: REVIEW PENDING (adversarial Opus review below once done).

## Answer in one paragraph
The continuity breaks at the **metric**. Along the path IDEEA -> ngref-fix, holding everything else fixed, the only single step that
moves strict TI from IDEEA's level down to ngref's level is replacing the identity metric by (G + lam I)^-1:
eucref-fix 0.403 -> ngref-fix 0.143 (-0.26 [-0.34,-0.18]) at the residual site, and the same sign at every other site and for every
direction source (head-site gradient -0.41, IDEEA's own clustered mean difference -0.11, ITI-like mean difference -0.06/-0.13, residual
mean difference -0.10). The metric's failure mode is the same everywhere: the refusal ("I have no comment") rate goes up. The other factors
do not break continuity: the reference set and IDEEA's train refit are neutral (+0.01), the activation context is neutral (+0.04), the
per-head vs global normalisation is neutral (-0.02/+0.02), the site matters little for the gradient (-0.04 n.s.), and the direction source
goes the OTHER way: replacing IDEEA/ITI's class-mean difference by the Euclidean gradient of the reference objective at IDEEA's own heads
raises strict TI (+0.19 to +0.23), to 0.46 vs IDEEA 0.31. "ngref is a continuous extension of IDEEA" is formally true for the
metric-on-mean-difference family (lam -> inf reproduces IDEEA bit-for-bit), but the extension direction is harmful: no finite lam helps
anywhere. Moreover IDEEA's direction is not an approximation of our gradient: at the same site the two are nearly orthogonal (cos 0.04).

## 1. What differs between IDEEA (paper cfg) and ngref-fix (code inspection, before running)
Positions identical (last position of every forward call). Seven differences, not five: site (24 o_proj-input heads over layers 5–25 vs
residual L13), direction source (class means vs grad of r = mean log p(correct refs) - mean log p(incorrect refs)), metric (I vs (G+lam I)^-1),
input dependence (per-head nc=2 clusters, min_perp selection vs none), normalisation (per head unit x naive_std x alpha vs global),
(vi) reference set (all refs of 395 dev q vs 316 train q minus 42 non-informative refs), (vii) activation context (last token of a bare
"Q: q"/"A: a" chat vs the eval prompt with instruction + few-shot, summed over answer positions).

## 2. Ladder (test, 395 q official / 150 q strict blind, matched dev KL 0.518; strict = mean of two blind Sonnet passes on FULL answers)
| arm | what it is | test KL | official TI | **strict TI [95% CI]** | strict T / I | refusal | dbl-neg | words |
|---|---|---|---|---|---|---|---|---|
| base | | 0 | .276 | .193 [.133,.260] | .87/.32 | .71 | .00 | 5.7 |
| L0 | IDEEA paper cfg (dev-fitted, all refs) | .514 | .572 | .310 [.243,.383] | .87/.40 | .31 | .01 | 18.9 |
| A_inf | IDEEA refit on train refs (plain) | .520 | .562 | .320 [.250,.393] | .90/.41 | .31 | .00 | 18.0 |
| A_10 | + block metric (lam_rel 10) | .548 | .418 | .213 [.153,.280] | .91/.29 | .64 | .00 | 11.1 |
| B_inf | ITI-like global mass-mean, per-head norm | .516 | .481 | .233 [.167,.300] | .87/.31 | .55 | .00 | 10.8 |
| B_10 | + block metric | .509 | .357 | .177 [.120,.237] | .91/.25 | .70 | .00 | 8.2 |
| E_inf | eval-context mean difference (head) | .513 | .486 | .273 [.207,.347] | .92/.34 | .52 | .01 | 12.2 |
| G_inf | mass-mean, global norm | .515 | .489 | .257 [.190,.327] | .90/.34 | .60 | .00 | 9.3 |
| G_10 | + full head metric G_H | .529 | .329 | .123 [.077,.177] | .93/.17 | .83 | .00 | 5.2 |
| F_euc0.05_inf | unpaired kernel arm (dev-best local, no metric) | .534 | .359 | .143 [.090,.203] | .84/.27 | .61 | .01 | 10.2 |
| C_inf | Euclidean grad of r at IDEEA's heads, per-head norm | .536 | .922 | **.460** [.383,.537] | .89/.51 | .00 | .10 | 31.0 |
| C_10 | + block metric | .525 | .808 | .410 [.337,.483] | .85/.47 | .20 | .15 | 20.7 |
| D_inf | clustered per-question gradient (non-IDEEA) | .529 | .924 | .430 [.353,.507] | .81/.50 | .01 | .10 | 31.0 |
| D_10 | + block metric | .528 | .828 | .393 [.320,.470] | .81/.48 | .19 | .13 | 21.3 |
| H_inf | Euclidean grad at heads, global norm | .523 | .906 | .440 [.367,.517] | .93/.48 | .00 | .04 | 30.3 |
| H_10 | + full head metric (natural gradient at IDEEA's site) | .565 | .175 | .027 [.007,.053] | .90/.05 | .81 | .12 | 6.1 |
| I_inf | residual L13 mass-mean (CAA-like, IDEEA rows) | .502 | .463 | .140 [.090,.193] | .84/.23 | .83 | .00 | 5.7 |
| I_10 | + residual metric | .481 | .205 | .043 [.013,.080] | .90/.10 | .83 | .00 | 4.6 |
| J_inf | residual eval-context mean difference | .436 | .200 | .050 [.020,.083] | .88/.12 | .82 | .00 | 7.3 |
| K_inf | eucref-fix (existing run, relabelled) | .523 | .848 | .403 [.330,.483] | .84/.47 | .12 | .15 | 22.2 |
| K_10 | ngref-fix (existing run, relabelled) | .462 | .415 | .143 [.090,.203] | .76/.22 | .61 | .20 | 10.3 |
| O1_b100 | optimised residual delta, contrastive J, KL leash | .428 | .223 | .047 [.020,.080] | .80/.13 | .76 | .04 | 5.5 |
| O2_b1 | optimised, correct-refs-only J | .534 | .600 | **.467** [.390,.543] | **.62**/.82 | .11 | .08 | 11.7 |
| O3_b10 | optimised, K=2 gated | .557 | .271 | .063 [.030,.103] | .84/.10 | .74 | .04 | 6.5 |
| O4_b1 | optimised, K=4 gated | .525 | .349 | .100 [.057,.150] | .78/.18 | .67 | .02 | 9.5 |
(metric member shown = lam_rel 10, the dev-best finite lam in every family; on official test TI lam_rel 10 is the best or tied-best finite lam in every family (A: .42/.42/.42 for 0.1/1/10; H: .13/.11/.18), see JSON.)
Strict labelling: 25 runs, 2592 unique (question, answer) items + 60 drift controls, two passes (different shuffles and chunks; 34 Sonnet
subagents), key outside the labelling dir; test-retest kappa truthful 0.89 / informative 0.86 / TI 0.86; drift controls vs the earlier
full-answer labels: TI 0.325 -> 0.317, agreement 0.95. Relabelled anchors reproduce the earlier strict numbers (base .193 = .193,
IDEEA .310 vs .300, eucref-fix .403 vs .413, ngref-fix .143 vs .150).

## 3. One-factor steps (paired over questions; strict 150 q; "break" = strict <= -0.07, CI excludes 0, official same sign)
| step | from -> to | official dTI | strict dTI [CI] | pass1 / pass2 | verdict |
|---|---|---|---|---|---|
| S1 data (dev all refs -> train filtered refs) | L0 -> A_inf | -.010 | +.010 [-.057,+.080] | .00/+.02 | neutral |
| S2 clustering off | A_inf -> B_inf | -.081 | -.087 [-.153,-.023] | -.10/-.07 | IDEEA's clustering helps (+0.09) |
| S3 context bare -> eval prompt | B_inf -> E_inf | +.005 | +.040 [-.020,+.103] | | neutral |
| S4 direction source md -> grad (eval ctx) | E_inf -> C_inf | +.435 | **+.187** [+.107,+.263] | +.21/+.17 | gradient better |
| S4b md -> grad (bare ctx) | B_inf -> C_inf | +.441 | **+.227** [+.143,+.310] | | gradient better |
| S5 normalisation per-head -> global (grad / md) | C->H, B->G | -.015 / +.008 | -.020 / +.023 | | neutral |
| **S6 metric, head site, grad (full G_H)** | H_inf -> H_10 | -.732 | **-.413** [-.493,-.333] | -.42/-.41 | **breaks** |
| S6 metric, head, grad (block) | C_inf -> C_10 | -.114 | -.050 [-.110,+.010] | | n.s. (same sign) |
| S6 metric, head, md (block) | B_inf -> B_10 | -.124 | -.057 [-.107,-.010] | | small, below threshold |
| **S6 metric, head, md (full G_H)** | G_inf -> G_10 | -.159 | **-.133** [-.197,-.073] | | **breaks** |
| **S6 metric on IDEEA itself** | A_inf -> A_10 | -.144 | **-.107** [-.170,-.047] | -.12/-.09 | **breaks** |
| S7 site head -> L13 (grad) | H_inf -> K_inf | -.058 | -.037 [-.103,+.037] | | n.s. |
| S7b site head -> L13 (md) | G_inf -> I_inf | -.025 | -.117 [-.180,-.057] | | md works worse at L13 |
| **S8 metric, residual, grad (eucref -> ngref)** | K_inf -> K_10 | -.433 | **-.260** [-.337,-.183] | -.26/-.26 | **breaks** |
| **S8b metric, residual, md** | I_inf -> I_10 | -.258 | **-.097** [-.150,-.043] | | **breaks** |
| S9 clustering on the gradient | C_inf -> D_inf | +.003 | -.030 [-.083,+.027] | | neutral |
| kernel (cluster-free, unpaired) vs IDEEA refit | A_inf -> F_euc0.05_inf | -.203 | -.177 [-.247,-.107] | | continuous kernel is worse |
| reverse, from ngref: site L13 -> heads | K_10 -> H_10 | -.241 | -.117 | | moving ngref to IDEEA's site makes it worse |
| reverse, from ngref: objective grad -> md | K_10 -> I_10 | -.210 | -.100 | | worse |
| reverse, from ngref: metric off | K_10 -> K_inf | +.433 | **+.260** | | the only fix |
| O2 (optimised, correct-only) vs IDEEA / eucref / ngref | | +.03 / -.25 / +.19 | +.157 / +.063 (n.s.) / +.323 | | best strict TI, but truthfulness .62 |

**Localisation verdict (pre-registered rule): the metric is the factor that breaks continuity**, at both sites and for both direction sources
(5 of 7 metric steps meet the break criterion; the 2 others have the same sign). No other single factor moves strict TI from IDEEA's
level (.31) to ngref's (.14); the direction source moves it the opposite way (up to .46). The prediction written in the prereg (metric)
holds. Pre-registered continuity-usefulness test: A_sel (finite lam) - A_inf = -0.107, CI lower bound -0.17 < -0.07 -> the extension
is formally continuous but not useful; the kernel arm F (tau, lam) is also worse than IDEEA (-0.18).

## 4. Why the metric hurts (what we can say)
- Dev and test show the same mechanism in every family: finite lam raises refusal (head grad 0.00 -> 0.81, residual grad 0.12 -> 0.61,
  IDEEA 0.31 -> 0.64, ITI-like 0.55 -> 0.70, CAA-like already 0.83). The gradient arms without metric almost never refuse.
- Not a first-order effect: the first-order change of log p("I have no comment.") per unit Fisher norm is not more positive for the metric
  directions (ngref lam_rel 0.1: -0.34/-0.28 train/val). (G+lam I)^-1 up-weights directions of LOW output sensitivity; at the step size
  needed to reach KL 0.5 these low-Fisher directions act non-linearly and the model falls back to its dominant mode (71% "I have no comment").
  This is a hypothesis consistent with the data, not a tested mechanism.
- The KL-leashed directly optimised contrastive arm (O1/O3/O4, the "first-principles" trust-region version; G is its quadratic expansion)
  hedges the same way (refusal .67–.76, strict .05–.10). Removing the incorrect-reference term (O2, likelihood of correct refs only,
  weak leash beta=1) removes the hedging (refusal .11) and gives the highest strict TI (.467), but by trading truthfulness (strict truth
  .62 vs .87 base) for informativeness (.82): it makes the model answer, and many answers are wrong.
- First-order objective gain is anti-correlated with generation quality here: the natural gradient has the largest first-order gain of r
  per unit Fisher norm (up to 4.1 at the head site vs 1.5 Euclidean vs 0.6 mass-mean), yet the worst strict TI.

## 5. Direction geometry (label-free, train refs)
| pair | head site (6144-d) | residual L13 |
|---|---|---|
| mass-mean vs Euclidean gradient | 0.04 | 0.07 |
| eval-ctx mean diff vs gradient | 0.07 | 0.07 |
| mass-mean bare vs eval ctx | 0.70 | 0.64 |
| gradient vs natural gradient (lam_rel 1) | 0.47 | 0.61 (lam 10: 0.88) |
| mass-mean vs metric mass-mean (lam_rel 1) | 0.93 (full G_H), 0.86 (block; per-head mean 0.91) | 0.88 |
| IDEEA cluster dir vs mass-mean | 0.78 | |
Sanity / limits (all pass): file-based IDEEA == IDEEA class; hard nearest-centroid kernel == IDEEA nearest_pos_neg; lam_rel 1e12 == plain for
A, B, I (40/40 identical generations); kernel tau = inf == pooled mass-mean (cos >= 1-7e-8). bf16 noise floor: lam_rel 1e6 (1e-5 relative
direction change) flips 4–13 of 40 greedy answers, so item-level differences between near-identical configs are noise.
Kernel effective neighbours (test prompts): Euclidean tau_rel 0.05 -> 36, 0.2 -> 433; Mahalanobis 0.05 -> 22, 0.2 -> 295 (of 1085/1288).

## 6. What is solid / what is not
Solid (large vs CI, both passes agree, official judges same sign):
1. The metric (G+lam I)^-1 lowers strict TI at every site and for both direction sources; at the residual site it accounts for the entire
   eucref -> ngref gap (-0.26). It works by raising refusals.
2. IDEEA's class-mean direction and our reference gradient are nearly orthogonal at the same site; IDEEA is not a special case of the
   gradient method except in the trivial lam -> inf limit of the metric-on-mean-difference family.
3. The Euclidean reference gradient at IDEEA's own heads beats IDEEA under strict labels (.46 vs .31, +0.15, same supervision class).
Not solid / caveats:
- Gradient arms have 4–15% double-negation answers and a large official-minus-strict gap (C_inf official .92 vs strict .46): they partly
  exploit the official judges; the strict gain survives but is half the official one. Single fit seed for the head-site gradient; one labeller
  family (Sonnet), test-retest agreement only.
- The metric was tested with ridge lam_rel in {0.1, 1, 10}; a much larger ridge interpolates to plain by construction, so "metric hurts"
  means "every amount of metric we could select on dev hurts", with lam_rel 10 the least harmful.
- Matched KL is approximate: realised test KL .43–.57 for the labelled runs (ngref .46, H_10 .565, O1 .43). O1_b10's matched strength was
  extrapolated (test KL 4.7, excluded from the strict set). For G/H/I no finite-lam config passed the dev refusal cap; their metric member is
  the max-dev-TI finite lam (declared deviation).
- O2's strict TI gain comes with a strict truthfulness drop to .62 (more false claims); it is not "more truthful".
- Tuning budgets: A–D, G–I 16 dev configs each, E/J 4, F 40, K 20, O 12 per variant, original IDEEA 59. Plain members won the dev selection
  in every family, so the budget asymmetry does not favour the metric arms.
- 25 strict runs: only the pre-declared steps are confirmatory; anything else is exploratory.

## Compute
GPU (A100 3g.20gb MIG): 18.1 GPU-h (smoke 0.6, failed OOM 0.8, features + fit + calibration 1.0, dev 5.3, arm O 8.2, test 2.1); <= 2 concurrent
jobs; official 7B judges only for evaluation, one at a time. 34 Sonnet labelling subagents.

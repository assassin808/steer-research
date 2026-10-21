# Pre-registration: IDEEA <-> ngref BRIDGE (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

Written 2026-10-20, **before any bridge run** (no bridge direction has been fitted, generated or judged). Includes the three
coordinator addenda (metric-on-mean-difference family; cluster-free kernel arm; correction: the kernel arm is UNPAIRED).
Code (to be written): `ideea_harness_gemma/scripts/bridge.py` (features / fit / calib / eval plans), `bridge_select.py`,
`bridge_report.py`, `bridge_blind_build.py`, `bridge_strict_merge.py`. Everything not stated here is exactly as in
`results/unified_gemma2b.md` and `results/ngref_fix_prereg.md`: IDEEA split (dev 395 = train 316 + val 79, test 395),
IDEEA eval prompt, greedy, 50 new tokens, bf16 generation, official truth/info 7B judges loaded one at a time (evaluation
only), realised per-token KL(steered||base) along the steered greedy answer.

## 0. Why, and what is actually different between IDEEA (paper cfg) and ngref-fix
Same-supervision result to explain (strict blind TI, 150 test q): IDEEA paper cfg 0.300, base 0.193, ngref-fix 0.150,
eucref-fix 0.413 (unconfirmed). Code inspection (before running) shows the two methods differ in MORE than the 5 listed factors:

| factor | IDEEA min_perp K24 a4 nc2 | ngref-fix (nfsum10, L13) |
|---|---|---|
| positions | last position of every forward call (prompt end, then each generated token) | identical (RowSteer "last") |
| site | inputs of o_proj for the top-24 heads (ITI probe ranking), spread over layers 5..25 (6144 dims) | residual stream output of decoder layer 13 (2304 dims) |
| supervision rows | all refs of the 395 DEV questions (train+val), incl. "I have no comment" | TRAIN (316 q) refs minus 42 non-informative ones |
| **context of the statistic** | activation at the LAST token of a bare chat `user: "Q: q" / assistant: "A: a"` (no instruction, no few-shot) | score features in the EVAL context (instruction + few-shot + "A: "), summed over all answer positions |
| direction source | per-head class-conditional means (K-means nc=2 per class, QAP matching), unit per cluster | grad of r = mean log p(correct) - mean log p(incorrect), per-question averaged |
| metric | identity | (G + lam I)^-1, G = within-question Fisher of 16 unlabeled base samples/train q |
| input dependence | per head, per step: min_perp picks the cluster dir with max abs(x . v) | none (one constant vector) |
| normalisation | per head: unit dir x naive_std_h x alpha (naive_std = std of head-activation norms) | global: c x mean residual norm |

So the ladder must also isolate (vi) the data/ref set and (vii) the activation context.

## 1. Sites, metrics, and the head-site Fisher
- Head site H = IDEEA's 24 top heads (ITI `sorted_idx`, unchanged), theta_H in R^{24x256} added to the o_proj INPUT at the
  last position of every forward call (IDEEA hook). In a teacher-forced pass over prompt+answer this is theta_H at positions
  P-1..P+T-2 (same as the residual protocol), so u_H = d log p(y)/d theta_H = sum_t d log p / d z_{l,h,t}.
- G_H = mean over TRAIN questions of the within-question covariance of u_H over 16 temperature-1 base samples (new samples,
  seed 0, unlabeled; same estimator as the residual G). Block G_hh = 256x256 diagonal block of head h.
- Relation to the residual form (stated, not assumed): for a single head h in layer l, o_proj is linear, but Gemma-2 applies
  `post_attention_layernorm` (RMSNorm) to the attention output before the residual add, so Delta resid_t = J_t W_o^{(l)} theta_h,
  J_t = RMSNorm Jacobian at position t. If J_t ~ J were position-independent, u_H = M^T u_R^{(l,mid)} with M = J W_o, and
  G_H = M^T G_R M (pull-back of the residual Fisher at the mid-layer-l residual), i.e. the head-site natural gradient is the
  residual natural gradient restricted to the image of M. Because J_t varies and the 24 heads span 13 layers, this is only an
  approximation; we never use it numerically — G_H is computed exactly by autograd at the head site.
- Residual site R = decoder-layer-13 output, G_R = the existing ngref-fix G (runs_ng/directions_fix_L13.npz).
- Ridge: lam = lam_rel x tr(G)/dim (dim = 256 for a block, 6144 for G_H, 2304 for G_R); lam_rel in {0.1, 1, 10, inf}; inf = no
  metric (plain). Sanity check (pre-registered, must pass before dev runs): lam_rel = 1e6 gives cos > 0.99999 with plain for
  every family and IDENTICAL greedy texts on the 40-question calibration set at one strength (families A and B).

## 2. Supervision (identical class for every new arm)
TRAIN-split (316 q) TruthfulQA reference answers with the 42 non-informative references removed (ngref-fix regex), no judge,
no judged generations, no val/test answers in any direction. VAL is used only through the dev selection rule (as before) and
the head ranking/naive_std that define IDEEA's site (kept fixed as part of the site). Mean-difference arms use the bare-context
answer-end activations (IDEEA's statistic) of these rows; gradient arms use r_fix ("sum"). L0 (original IDEEA, dev-fitted, all
refs) is kept as the anchor; L0 -> A_inf isolates the ref set/data factor.

## 3. Arms (all at matched realised KL; strength = the only within-config continuous knob)
Mean-difference (IDEEA-source) arms — POOLED, UNPAIRED class means (no per-question pairing):
- **A  IDEEA clustered + metric** (head site, per-head norm): IDEEA's fit_fixed_nc (nc=2, K-means on positive and on negative
  rows separately, QAP matching) on the train rows; cluster direction v_{h,c}(lam) = unit((G_hh + lam I)^-1 (mu+_{h,sigma(c)} - mu-_{h,c}));
  min_perp selection with these directions (IDEEA code path, directions file substituted). A_inf = IDEEA refit on train refs.
- **B  ITI-like global + metric** (head, per-head norm): v_h(lam) = unit((G_hh + lam I)^-1 (mean pos - mean neg)_h) x naive_std_h x alpha
  (IDEEA's ITI code path). B_inf = ITI K=24 refit on train refs.
- **G  ITI-like global + FULL head metric** (head, global norm): v = (G_H + lam I)^-1 Delta mu_H (6144-d), theta = alpha x S x v/||v||,
  S = sqrt(sum_h naive_std_h^2) (so equal alpha = equal total norm as per-head). G_inf = mass-mean with global normalisation.
- **F  cluster-free kernel arm (continuous IDEEA)** (head, per-head norm, per step like IDEEA): for current head input x,
  mu+(x) = sum_i k(x,p_i) p_i / sum_i k(x,p_i) over ALL positive train rows p_i, mu-(x) likewise over negatives n_j,
  k(x,y) = exp(-dist(x,y)^2 / tau), v_h(x) = unit((G_hh + lam I)^-1 (mu+(x) - mu-(x))) x naive_std_h x alpha.
  dist: Euclidean, or Mahalanobis with M_h = G_hh + 0.1 tr(G_hh)/256 I. tau = tau_rel x median pairwise dist^2 between train rows of
  that head (per head, per dist). Grid: tau_rel in {0.05, 0.2} x dist {euc, mah} plus tau = inf (dist irrelevant) = 5 tau points
  x lam_rel in {1, inf} = 10 configs. Limits checked in code before dev runs: tau = inf equals B_inf (pooled mass-mean) exactly;
  hard nearest-K-means-centroid weights (one-hot on IDEEA's real nc=4 dev-fitted centroids) reproduce IDEEA nearest_pos_neg
  generations exactly (40 calib q, alpha 4); lam = inf removes the metric. Reported: effective neighbours n_eff = 1/sum w^2
  (normalised w, per head, per class) at the prompt-end activation of every test question.
- **E  eval-context mean difference** (head, per-head norm, lam = inf only): pooled mean over kept correct refs of the
  answer-position-averaged head input in the EVAL context minus same for incorrect refs (isolates the context factor vs B_inf).
- **I  residual CAA-like + metric** (L13, global norm c x hnorm): Delta mu at L13 last token of the bare contrast chats
  (IDEEA's rows; NOT IDEEA's CAA, which uses A/B multiple-choice prompts), v(lam) = (G_R + lam I)^-1 Delta mu. I_inf = plain.
- **J  residual eval-context mean difference** (L13, lam = inf only), as E at the residual site.
Gradient (our objective) arms:
- **C  head gradient, per-head norm**: v_h(lam) = unit(((G_hh + lam I)^-1 g_H)_h) block metric. C_inf = Euclidean grad of r_fix at H.
- **H  head gradient, global norm, full metric**: v(lam) = (G_H + lam I)^-1 g_H. H_inf = Euclidean (eucref at the head site).
- **D  clustered gradient** (explicitly NON-IDEEA: per-question gradients g_q are paired by construction): per head, K-means
  (nc=2, seed 0) on the eval-prompt-end head inputs of the train questions; direction of cluster c = unit((G_hh + lam I)^-1 mean_{q in c} g_{q,h});
  per-step selection by nearest context centroid; per-head norm.
- **K  residual gradient** = existing eucref-fix (K_inf, `fx:efsum:13`) and ngref-fix (`fx:nfsum{0.1,1,10}:13`); existing dev runs reused.

## 4. Dev-only tuning (rule of ngref_fix_prereg.md §3, unchanged)
- Strength: label-free KL scan on the first 40 dev questions (head arms alpha in {0.5,1,2,4,8,16}; residual c in {0.05,0.1,0.2,0.4,0.8,1.6}),
  log-log interpolation to dev target KLs {0.1, 0.3, 0.6, 1.0}; each target run on all 395 dev questions, official judges.
- Config counts: A, B, C, D, G, H, I: 4 lam x 4 KL = 16 each; E, J: 4 each; F: 10 x 4 = 40; K: existing 20 (3 lam x 4 + eucref 8);
  original IDEEA/ITI grid 59/14 (existing). Families with lam/tau contain their plain member, so selection can return "no metric".
- Rule per family: among dev configs with dev KL <= 1.2 and dev refusal <= 0.722, max official dev TI, ties (0.005) -> lower KL.
  "sel" = selected config; if it is the plain (lam = inf) member, "sel" for the metric comparison = best FINITE-lam config by the same rule.
  For F, "sel" = selected (tau, lam); also reported: best (tau finite, lam inf) and best (tau inf, lam 1).
- Matched strength = strength giving dev KL 0.518 (IDEEA paper cfg's dev KL), interpolated from that config's 4-point dev KL sweep.

## 5. Test (each once, after selection, no changes)
Official judges, 395 test q, matched strength: every lam of A, B, C, D, G, H, I (28 runs), E, J, F-sel, F(tau inf, lam inf),
F best-local-plain, F best-(inf, 1), K: nfsum0.1/1 matched (new), eucref-fix / ngref-fix matched (existing).
Strict blind labels (150 test q of the earlier strict set, seed 0): base, L0 (original IDEEA), A_inf, A_sel, B_inf, B_sel, C_inf, C_sel,
D_inf, D_sel, E, F_sel, G_inf, G_sel, H_inf, H_sel, I_inf, I_sel, J, K_inf (eucref-fix), K_sel (ngref-fix nfsum10) = 21 runs,
relabelled together in ONE pool (base/L0/eucref/ngref relabelled, not reused), full untruncated parsed answers (no clean()), items
deduplicated by (question, answer), shuffled, random ids, key outside the labelling dir, 2 passes with different shuffles/chunking,
Sonnet subagents with the archived TruthfulQA-definition prompt (labeller_prompt_fix.txt), plus 60 drift-control items from the
earlier full-answer set. Item score = mean of the two passes.

## 6. Metrics
Official T/I/TI (paired bootstrap over questions, 2000), strict T/I/TI with CI, refusal rate (is_refusal regex), mean words,
double-negation rate, 50-token-limit rate, realised test KL. Label-free: cosines between directions at the same site (head 6144-d
global vectors and mean per-head cosine; residual 2304-d): Delta mu (bare), Delta mu (eval ctx), g, (G+lam I)^-1 g, (G+lam I)^-1 Delta mu,
IDEEA cluster dirs vs global; first-order val gain of r_fix per unit G-norm for each direction; n_eff for F.

## 7. Analyses and what each outcome means (decided now)
Steps (paired strict and official Delta TI at matched KL, 150 / 395 q):
S1 data: L0 -> A_inf. S2 clustering: A_inf -> B_inf. S3 context: B_inf -> E. S4 objective (md -> grad, same ctx, head, per-head): E -> C_inf
(and B_inf -> C_inf). S5 normalisation: C_inf -> H_inf (grad), B_inf -> G_inf (md). S6 metric @ head: H_inf -> H_sel, C_inf -> C_sel,
B_inf -> B_sel, G_inf -> G_sel, A_inf -> A_sel. S7 site (Euclidean grad): H_inf -> K_inf; (md): G_inf -> I_inf. S8 metric @ residual:
K_inf -> K_sel (= eucref-fix -> ngref-fix), I_inf -> I_sel. S9 clustering on the gradient: C -> D. Reverse direction = the same steps
read from ngref (K_sel) back to IDEEA, plus single-factor toggles from each end (metric on IDEEA = A_sel vs A_inf; site of ngref = H_sel vs K_sel;
objective of ngref = I_sel vs K_sel; metric of ngref = K_inf vs K_sel).
- A step "breaks continuity" if its strict Delta TI <= -0.07 with paired 95% CI excluding 0 AND official Delta TI has the same sign.
  Power: paired strict SE ~0.035, so |Delta| < ~0.07 is "no detectable effect", not "no effect".
- **Localisation verdict**: the factor(s) whose step(s) break continuity on the IDEEA -> ngref path and NOT on the other steps.
  Prediction written now (from existing numbers: eucref-fix 0.413 > IDEEA 0.300 > ngref-fix 0.150): the break is the METRIC
  (S6/S8 negative), not site/objective/clustering. If S6 is ~0 at the head site but S8 negative, the metric breaks only at the residual site.
- **Continuity claim** ("our method is a continuous extension of IDEEA"): formally true for families A/B/F/G/I (lam -> inf recovers the
  plain method, verified). It is *useful* only if a dev-selected finite-lam member is not worse than its plain member: strict
  Delta(A_sel - A_inf) and Delta(F_sel - A_inf-like plain) CI lower bound > -0.07. If finite lam is significantly worse at every site,
  the extension exists but the metric direction is harmful for generation — report that as the answer.
- If no step breaks continuity but the end points differ significantly, the effect is distributed (several small steps) — report so.

## 8. Honest-risk notes written in advance
- 21 strict runs x 150 q; multiple comparisons: only the pre-declared steps above are confirmatory; anything else is exploratory.
- Strict labeller = one model family (Sonnet), test-retest only.
- The head Fisher uses fresh samples (not those of the residual G); G_H (6144^2) from 5056 samples is rank-deficient: the ridge grid covers this.
- The F kernel arm is new code with per-step kernel evaluation; its exact limits are verified in code, but its tau grid is coarse.

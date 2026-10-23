# IDEEA <-> ngref bridge (Gemma-2-2B-it, IDEEA official TruthfulQA protocol)

Date 2026-10-23. Pre-registration: `results/bridge_prereg.md` (commits 13312ad + amendment de288ba, written before any run).
Numbers: `results/bridge.json`. Code: `ideea_harness_gemma/scripts/bridge.py` (features, fit, hooks, arm O), `bridge_select.py`,
`bridge_testspecs.py`, `bridge_report.py`, `bridge_blind_build.py`, `bridge_strict_merge.py`, `br_*.sbatch`.
Raw generations, labels, blind key: `~/scratch/ideea_gemma/{runs_ng/bridge, results_gm/*bridge*, blind_bridge, blind_bridge_keys}` (not in git).


Status: adversarially reviewed (Opus, Section 7); text below is the post-review version.

## Answer in one paragraph
The largest break in continuity is the **metric**. Holding everything else fixed, replacing the identity by (G + lam I)^-1 (lam_rel in
the pre-registered grid {0.1, 1, 10}) lowers strict TI at both sites and for both direction sources: eucref-fix 0.403 -> ngref-fix 0.143
(-0.26 [-0.34,-0.18]), Euclidean head-site gradient 0.44 -> natural gradient 0.03 (-0.41), IDEEA's own clustered mean difference 0.32 -> 0.21
(-0.11), global mass-mean -0.13, residual mass-mean -0.10 (5 of 7 metric steps meet the pre-registered break criterion; the other 2 have the
same sign). By the pre-registered rule two other steps also qualify as breaks, both smaller: turning IDEEA's clustering off (-0.09) and moving
the mass-mean from the heads to residual L13 (-0.12). The direction source goes the OTHER way: replacing the class-mean difference by the
Euclidean gradient of the reference objective at IDEEA's own heads raises strict TI by +0.19..+0.23 (0.46 vs IDEEA 0.31). Data/reference set,
activation context and normalisation are neutral. **Mechanism (verified to first order, label-free):** the top eigenvector of the Fisher G
is the base model's "I have no comment" vs "answer" axis (cos 0.96 head site, 0.98 L13; 67-71% of base samples start with the abstention),
and the plain directions work largely by pushing the model off that axis; (G + lam I)^-1 divides exactly this component away (refusal log-odds
per unit Fisher norm: eucref -1.50 -> ngref -0.39; head grad -1.35 -> -0.28; mass-mean -0.53 -> +0.08), so metric directions do not reduce
abstention and spend their KL budget elsewhere. The failure is therefore a property of a Fisher estimated on an abstention-dominated base
distribution under this prompt, not necessarily of the metric idea itself. "ngref is a continuous extension of IDEEA" is formally true only
for the metric-on-mean-difference family (lam -> inf reproduces the train-refit IDEEA exactly); within the tested lam range the extension is
harmful, and IDEEA's direction is not an approximation of our gradient (cos 0.04 at the same site).

## 1. What differs between IDEEA (paper cfg) and ngref-fix (code inspection, before running)
Positions identical (last position of every forward call). Seven differences, not five: site (24 o_proj-input heads over layers 5–25 vs
residual L13), direction source (class means vs grad of r = mean log p(correct refs) - mean log p(incorrect refs)), metric (I vs (G+lam I)^-1),
input dependence (per-head nc=2 clusters, min_perp selection vs none), normalisation (per head unit x naive_std x alpha vs global),
(vi) reference set (all refs of 395 dev q vs 316 train q minus 42 non-informative refs), (vii) activation context (last token of a bare
"Q: q"/"A: a" chat vs the eval prompt with instruction + few-shot, summed over answer positions).

## 2. Ladder (test, 395 q official / 150 q strict blind, matched dev KL 0.518; strict = mean of two blind Sonnet passes on FULL answers)
| arm | what it is | test KL | official TI | **strict TI [95% CI]** | strict T / I | abst.-prefix | dbl-neg | words* |
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
("abst.-prefix" = answer starts with an abstention after clean(), i.e. the text before the first blank line; with the full text the rate is far lower for long-answer arms, e.g. L0 .03, G_10 .26 — metric arms produce "I have no comment.\n\n<hedge>". words* also on clean() text. Selection is unaffected (same definition as the prereg rule). Metric member shown = lam_rel 10, the dev-best finite lam in every family; on official test TI lam_rel 10 is the best or tied-best finite lam in every family (A: .42/.42/.42 for 0.1/1/10; H: .13/.11/.18), see JSON.)
Strict labelling: 25 runs, 2592 unique (question, answer) items + 60 drift controls, two passes (different shuffles and chunks; 34 Sonnet
subagents), key outside the labelling dir; test-retest kappa truthful 0.89 / informative 0.86 / TI 0.86; drift controls vs the earlier
full-answer labels: TI 0.325 -> 0.317, agreement 0.95. Relabelled anchors reproduce the earlier strict numbers (base .193 = .193,
IDEEA .310 vs .300, eucref-fix .403 vs .413, ngref-fix .143 vs .150).

## 3. One-factor steps (paired over questions; strict 150 q; "break" = strict <= -0.07, CI excludes 0, official same sign)
| step | from -> to | official dTI | strict dTI [CI] | pass1 / pass2 | verdict |
|---|---|---|---|---|---|
| S1 data (dev all refs -> train filtered refs) | L0 -> A_inf | -.010 | +.010 [-.057,+.080] | .00/+.02 | neutral |
| S2 clustering off | A_inf -> B_inf | -.081 | -.087 [-.153,-.023] | -.10/-.07 | **break** (IDEEA clustering helps +0.09) |
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
| S7b site head -> L13 (md) | G_inf -> I_inf | -.025 | -.117 [-.180,-.057] | | **break** (off-path; md worse at L13) |
| **S8 metric, residual, grad (eucref -> ngref)** | K_inf -> K_10 | -.433 | **-.260** [-.337,-.183] | -.26/-.26 | **breaks** |
| **S8b metric, residual, md** | I_inf -> I_10 | -.258 | **-.097** [-.150,-.043] | | **breaks** |
| S9 clustering on the gradient | C_inf -> D_inf | +.003 | -.030 [-.083,+.027] | | neutral |
| kernel (cluster-free, unpaired) vs IDEEA refit | A_inf -> F_euc0.05_inf | -.203 | -.177 [-.247,-.107] | | continuous kernel is worse |
| reverse, from ngref: site L13 -> heads | K_10 -> H_10 | -.241 | -.117 | | moving ngref to IDEEA's site makes it worse |
| reverse, from ngref: objective grad -> md | K_10 -> I_10 | -.210 | -.100 | | worse |
| reverse, from ngref: metric off | K_10 -> K_inf | +.433 | **+.260** | | the only fix |
| O2 (optimised, correct-only) vs IDEEA / eucref / ngref | | +.03 / -.25 / +.19 | +.157 / +.063 (n.s.) / +.323 | | best strict TI, but truthfulness .62 |

**Localisation verdict (pre-registered rule: break = strict <= -0.07, CI excludes 0, official same sign).** Breaks: the metric (S6 full-G
head grad -0.41, S8 residual grad -0.26, S6 full-G md -0.13, S6 IDEEA -0.11, S8b residual md -0.10; block-metric C -0.05 n.s., B -0.06 below
threshold), clustering off (S2 -0.09) and moving the mass-mean to L13 (S7b -0.12). Along the actual path L0 -> A -> B -> E -> C -> H -> K_inf -> K_10
there are two breaks (S2 -0.09, S8 -0.26) and one large improvement (S4, direction source, +0.19/+0.23). So the metric is the largest break and the
only one that moves strict TI from IDEEA's level (.31) to ngref's (.14); the pre-registered "breaks only on the path and nowhere else" clause is NOT
met (S7b is off-path). The pre-registered prediction (metric) holds as the dominant factor. Caveat: S8 re-labels the two runs that motivated the
hypothesis (same 150 q); the new confirmatory evidence is S6 at IDEEA's site (H, G, A). Continuity-usefulness test: A_10 - A_inf = -0.107, CI lower
bound -0.17 < -0.07 -> within lam_rel <= 10 the extension is not useful; the cluster-free kernel arm is also worse than IDEEA (-0.18; note dev selected
its tau = inf member, i.e. plain mass-mean; F_euc0.05_inf is the best finite-tau member, an undeclared substitution for "F_sel").

## 4. Why the metric hurts (post-review; scripts `bridge_mech_head.py`, `bridge_mech_resid.py`, outputs in ~/scratch logs)
- The dominant eigenvector of G is the abstain-vs-answer axis: within-question score difference (abstention-prefixed minus other base samples)
  has cos 0.962 with G_H's top eigenvector (95% of that axis' energy in the top 5) and 0.976 with G_R's. 67% (head samples) / 71% (L13 samples)
  of unlabeled base samples start with "I have no comment".
- First-order change of the abstain-vs-answer log-odds per unit Fisher norm (negative = away from abstaining): head grad H_inf -1.35 -> H_10 -0.28;
  C -1.25 -> -0.71; mass-mean G -0.53 -> +0.08, B -0.60 -> -0.10; residual eucref -1.50 -> ngref(10) -0.39 -> (1) -0.15 -> (0.1) +0.01.
  The plain directions' gains come largely from suppressing abstention; the metric removes that component because it is the highest-Fisher
  (most KL-expensive) direction. This explains why metric arms keep the abstention-prefix rate near base (.61-.83 vs .71) while plain arms cut it.
  (The draft's single-string "dlogp(I have no comment.)" statistic was the wrong quantity and was quoted selectively; withdrawn.)
- Consequence: the failure is specific to a Fisher estimated on an abstention-dominated base distribution under IDEEA's "reply 'I have no
  comment' unless certain" prompt. It does not show that a behavioural metric is wrong in general.
- lam range: lam_rel 10 is still a strong metric (lam = 10 tr(G)/d is ~1/65 (head) and ~1/19 (L13) of G's top eigenvalue), so the transition
  region lam ~ top eigenvalue (lam_rel ~ 100-1000) was NOT tested; "no finite lam helps" holds only for lam_rel <= 10.
- Metric x normalisation interact: with per-head normalisation the block metric's effect is small (C -0.05 n.s., B -0.06); with global
  normalisation and the full G_H it is large (H -0.41, G -0.13). H_10 also spreads the norm over more heads (effective heads 9.3 -> 17.4) and
  puts up to 33 x naive_std on single heads (H_inf 12.9, IDEEA 4).
- Arm O (directly optimised, KL leash) is EXPLORATORY only: the optimiser was badly scaled (Adam lr 3e-3 on z with delta = hnorm z gives
  ~0.14 hnorm per step; val KL 1-8 nats/token at the first checkpoint for O3/O4/O1_b10; final deltas rescaled to KL 0.5, which voids the beta
  leash). O1/O3/O4 hedge (abstention-prefix .67-.76, strict .05-.10). O2 (correct-only, beta 1) gives the highest strict TI (.467) but strict
  truthfulness .62 (base .87): it makes the model answer and many answers are false; O2 with beta 10/100 hedges again (dev abst. .55/.73), so
  "removing the incorrect-ref term removes hedging" is confounded with beta. No claim about the trust-region formulation is made.

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
Solid (CIs far from 0, both label passes agree, official judges same sign, no leakage, KL mismatch negligible — dev TI is flat in KL .45-1.2 for K_10/H_10):
1. Within lam_rel <= 10 the metric (G+lam I)^-1 lowers strict TI at both sites and for both direction sources (largest single break), and the
   first-order mechanism is identified: G's top eigenvector is the abstain-vs-answer axis, which the metric divides away.
2. IDEEA's class-mean direction and our reference gradient are nearly orthogonal at the same site; IDEEA is not a special case of the gradient
   method, only the trivial lam -> inf limit of the metric-on-mean-difference family.
3. The Euclidean reference gradient at IDEEA's own heads beats IDEEA under strict labels (.46 vs .31) in this pipeline.
Not solid / caveats:
- The labeller sees the reference answers, and the gradient arms are trained on the likelihood of train references (rule 6): the S4 advantage
  could partly be a style preference of the labeller; it does not affect the metric steps (same objective on both sides). Gradient arms also have
  4-15% double-negation answers (items with double negation: strict TI .32 vs .47 without) and a large official-minus-strict gap (C_inf .92 vs .46).
- One fit seed for the head Fisher (the earlier ngref-fix seed variance on official TI was .415/.420/.491); one labeller family, test-retest only;
  ~24 uncorrected 95% CIs (S2, p ~ .006, would not survive Bonferroni; the metric steps H, K, G would).
- Pre-registered sanity check "lam_rel 1e6 gives identical texts" FAILED (4-13/40 answers differ from bf16 rounding); the threshold was changed to
  1e12 (identical). "Reproduces IDEEA" means the train refit A_inf (cos .54 with the dev-fitted L0 cluster dirs; S1 is neutral), verified on 40 dev q.
- Matched KL approximate (labelled runs .43-.57); O1_b10 extrapolated (KL 4.7, excluded). For G/H/I no finite lam passed the dev refusal cap;
  the metric member is lam_rel 10, which is also their max-dev-TI finite lam (declared deviation). The reverse step K_10 -> H_10 is not a pure
  site change (lam_rel 10 is a different relative strength at the two sites).
- Tuning budgets: A-D, G-I 16 dev configs each, E/J 4, F 40, K 20, O 12 per variant, original IDEEA 59; plain members won dev selection in every
  family, so the budget does not favour the metric arms.
Next (not run): lam_rel in {100, 1000} (the transition region) as a new pre-registered item; a Fisher estimated on non-abstaining samples or with
the abstain axis projected out (tests whether the metric helps once the abstention mode is removed); a second labeller family; head-Fisher seeds.

## 7. Adversarial review (Opus) — summary of findings and what was changed
No leakage (splits disjoint; fit on train refs/unlabeled samples; selection on dev only; test strengths from dev sweeps). Blind set clean (2592
items, chunk answers == full generations for all 3750 run-items, 0 mismatches; key outside labelling dir). Strict item arrays reproduced from the
key; hooks implement the described families (file-substituted IDEEA/ITI directions, global-norm encoding, block vs full metric, D's dummy
centroids, kernel limits, GateSteer train/generate equivalence, KL(p_delta||p_0)). Key steps recomputed (10k bootstrap) identical. Changes made:
(1) verdict rewritten to follow the pre-registered rule (S2 and S7b are also breaks; metric is the largest, not the only one); (2) "no finite lam
helps" restricted to lam_rel <= 10; (3) mechanism paragraph replaced by the verified first-order abstain-axis mechanism (the draft's statistic was
wrong and selectively quoted); (4) arm O downgraded to exploratory (optimiser scaling); (5) S8 flagged as re-labelling of the motivating runs;
(6) metric x normalisation interaction stated; (7) "refusal" renamed abstention-prefix rate with full-text caveat; (8) failed 1e6 sanity check
and the F_sel substitution declared; (9) labeller-sees-references caveat for S4.

## Compute
GPU (A100 3g.20gb MIG): 18.1 GPU-h (CPU-only for the review/mechanism checks) (smoke 0.6, failed OOM 0.8, features + fit + calibration 1.0, dev 5.3, arm O 8.2, test 2.1); <= 2 concurrent
jobs; official 7B judges only for evaluation, one at a time. 34 Sonnet labelling subagents.

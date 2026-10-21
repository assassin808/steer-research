# PoC v3.1: trait score as a function of KL arc length (2026-10-21)

Pre-registration: `poc_v31_design.md` (commit 90cfd69, before any run; unchanged). Questions: `poc_v31_questions.json`
(172 fresh Claude-written questions, disjoint from v1-v3 by exact string). Code: ~/steer_poc/v31/{v31.py, analyze.py}.
Raw outputs (not committed): ~/steer_poc/v31/out/. Summary numbers: `poc_v31_summary.json`.
Jobs (Narval a100_3g.20gb): smoke 6592056 (14 min), sycophantic 6592599 (36 min), optimistic 6597187 (26 min),
A1/A2/A3 6598449 (37 min), evil 6598975 (13 min). Total **2.1 GPU-hours**.

Setup in one paragraph. Qwen3-1.7B generates (T=1, 128 new tokens) and judges. STEER adds c·u at all positions of
layer 14 (evil: 20), u = unit mean of the v3 per-paraphrase persona vectors, for two disjoint paraphrase sets
A = {0,1,2} and B = {3,4} (cos(u_A, u_B) = 0.76). The coordinate is s = per-token KL arc length, i.e. the sum of
sqrt(2·KL) between consecutive c on a 0.5R DEV grid (R = median |v_p|). The cap is fluency on DEV; six levels are equally
spaced in s up to s_max = min over sets. TEST uses 30 own-trait questions (sycophancy: the user states an opinion)
× 20 samples per cell. The primary judge is a thresholded 3-option Qwen judge, averaged over both option orders,
score = P(+pole) − P(−pole) in [−1, 1]. Reference: 180 blind items per trait (150 for evil), two independent
Sonnet passes, shuffled ids, key outside the labelling dir.

## Verdict (pre-registered criteria)

| trait | set | K0 Qwen≈Claude ρ≥0.6 | K1 d≥0.5 (Qwen / Claude) | K2 per-q slope>0 vs oracle−0.10 | K3 ρ ≥ 0.8·oracle | K4 TEST steps even, KL₀↑ | K5 fluent | K6 sets agree ≤0.25 sd | verdict |
|---|---|---|---|---|---|---|---|---|---|
| sycophantic | A | ✓ 0.85 | ✓ 0.73 / 0.72 | ✓ 0.93 vs 0.85 | ✓ 0.261 vs 0.259 | ✓ | ✓ | ✓ 0.09 | **PASS** |
| sycophantic | B | ✓ | ✓ 0.63 / (pooled) | ✓ 0.93 vs 0.84 | ✓ 0.230 vs 0.227 | ✓ | ✓ | ✓ | **PASS** |
| optimistic | A | ✓ 0.75 | ✓ 0.78 / 1.45 | ✓ 0.87 vs 0.83 | ✓ 0.316 vs 0.312 | ✓ | ✓ | ✓ 0.16 | **PASS** |
| optimistic | B | ✓ | ✓ 0.82 / (pooled) | ✓ 0.80 vs 0.79 | ✓ 0.340 vs 0.321 | ✓ | ✓ | ✓ | **PASS** |

- **The Claude K1 check pools both sets.** It uses 22 blind items per level, with levels 1-5 split across the two sets.
  Claude d (level 5 vs level 0) is 0.72 for sycophancy (95% CI of the difference on the 1-5 scale: [0.18, 1.73]) and
  1.45 for optimism ([0.66, 1.59]).
- **Claude test-retest** (Spearman between the two passes) is 0.96 for sycophancy and 0.83 for optimism.
- **K0 is computed on all 180 blind items**, including the 30 A1 grid items. W1-only values are 0.85 and 0.77;
  steered-only 0.87 and 0.74; within-level pooled 0.86 and 0.71.
- **Label alignment.** The reviewer found that labellers hand-typed ordered label arrays, and that a block of optimism
  pass 2 had low test-retest. A fresh optimism pass 3 (new shuffle, explicit id→label pairs) gives test-retest
  0.885 with pass 1 and 0.848 with pass 2. K0 is 0.789 for passes 1+3 against 0.755 for 1+2, and Claude d is 1.52
  against 1.45. The verdict is unchanged. On the W1 items alone, the secondary v3
  0-9 digit judge agrees much less: ρ = 0.58 (sycophancy) and 0.60 (optimism). The thresholded judge is the fix that
  v3 needed.
- **Evil** (on/off check, no criteria) is reported below.

**So, for sycophancy and optimism, realised KL arc length along a fixed norm-normalised vector passes our
pre-registered standard for an intensity coordinate.** The mean trait score rises with s, the questions are fresh, both
vector sets reproduce, and the Claude reference confirms the judge and the endpoint effect. This is what v3 could not show.

Three limits from the review:
- **T(s) is a population rate of near-binary verdicts, not a graded per-answer intensity.** 70% (sycophancy) and 86%
  (optimism) of item scores have |score| > 0.9. On steered blind items, Claude's Spearman(level, label) is only 0.18
  for sycophancy and 0.48 for optimism.
- **K2 and K3 cannot distinguish path shapes.** Step-shaped mixtures of the real endpoint samples also pass, unless the
  step sits at level 1 or level 5. They confirm a positive trend of the right size, not linearity. Shape evidence comes
  only from the level means.
- **Claude confirms the endpoints, not the shape.** Its sycophancy level means are non-monotone, and level 3 (2.93)
  sits below level 0 (3.09).

## W1 (CONFIRMATORY): T(s) — `poc_v31_W1_trait_vs_s.png`

The table gives the Qwen score mean at s_k, k = 0..5:

| | s_max | set A | set B | prompted reference (score, chord s) |
|---|---|---|---|---|
| sycophantic | 1.18 | 0.32, 0.42, 0.52, 0.61, 0.71, 0.77 | 0.32, 0.41, 0.52, 0.58, 0.67, 0.71 | pos 0.85 @ 1.03; neg 0.00 @ 0.61 |
| optimistic | 1.49 | 0.15, 0.33, 0.50, 0.63, 0.69, 0.74 | 0.15, 0.40, 0.63, 0.70, 0.76, 0.76 | pos 0.85 @ 0.99; neg −0.53 @ 1.11 |

- **Shape.** T(s) is close to linear in s for sycophancy. For optimism it is concave: it saturates near the pos-prompt
  level, and set B saturates earlier.
  - At equal s, optimism set B moves the trait faster at s ≤ 0.6, by up to 0.12, i.e. 0.16 sd (still within K6).
  - So the same KL budget buys slightly different trait gains along different vectors. This is the expected
    "efficiency" variation, and it is small here.
- **The coordinate transfers from DEV to TEST almost exactly.**
  - TEST step distances sqrt(2·KL) are 0.23-0.25 for sycophancy and 0.27-0.30 for optimism. Every step is within
    ±7% of the median; the criterion allowed [0.67, 1.5].
  - Chord sqrt(2·KL(p_k‖p_0)) vs summed TEST arc length at level 5 (exploratory):
    - sycophancy: 1.206 vs 1.206 (A), 1.233 vs 1.213 (B);
    - optimism: 1.350 vs 1.432 (A, −6%), 1.231 vs 1.401 (B, −12%), with the gap growing with k.

    This shows only that per-token KL from the origin grows roughly quadratically along this 1-d path, less so for
    optimism. It is not evidence about curvature.
  - Sequence-level arc length (pre-registered secondary) at s_max is 13.1 nats^½ for sycophancy (both sets) and 16.9
    for optimism. Sequence KL(p_5‖p_0) is 88 and 97-117 nats.
- **The fluency cap binds through neutral-model NLL**, not through coherence.
  - Median NLL rises from 0.43 to 1.25-1.45 nats per token against a limit of 1.60-1.66.
  - Coherence stays between 5.7 and 6.9.
- **Length confound, sycophancy only.** The share of responses that end before the 128-token cap rises from 0% to
  17% (set A) and 28% (set B) at s_max, through 0/1/1/3/9/17% (A) and 0/1/1/5/14/28% (B) per level; mean length is 122
  and 119 tokens. Optimism responses always hit the cap. Capped-only responses still go 0.32 → 0.73 (set A), so length
  does not drive the trend. However, set B's last step is length-linked: capped-only 0.65 → 0.62, while natural-end
  items score 0.94.
- **Heterogeneity across questions is large.**
  - Per-question curves range from flat to +1.5 over the s range (thin lines in the figure).
  - Mean per-question slope differs by cluster (KMeans(4) of the L14 neutral prompt state): sycophancy 0.26-0.46 per
    unit s, optimism 0.0-0.46. This is exploratory; there are about 7 questions per cluster and no test was run.
  - The flat optimism cluster (slope 0.00) consists of questions already scored near +1 at s = 0, i.e. a ceiling.
- **K2 and K3 sit at the oracle reference.** Pooled ρ is 0.23-0.34 for both, and the per-q positive-slope share is
  0.80-0.93 against 0.79-0.85. As noted above, the reference is matched by step-shaped paths too, so this says
  "positive trend of the expected size" and nothing about shape. Pooled ρ is low because single answers are
  near-binary and noisy (sd 0.6-0.75 on [−1, 1]).

**Evil on/off check** (`poc_v31_W1_evil_on_off.png`, L20, 16 questions, s_max = 1.42):

| | level 0 | level 1 | level 2 | level 3 | level 4 | level 5 |
|---|---|---|---|---|---|---|
| Qwen P(persona on) | 0 | 0 | 0 | 0.003 | 0.028 | 0.087 [0.04, 0.13] |
| Claude P(on) (22 items per level) | 0 | 0 | 0 | 0.05 | 0.05 | 0.18 |

- The pos-prompt reference sits at P(on) 0.31 (Claude 0.22) at chord s = 1.19.
- Within the fluent range, steering along the evil vector switches the persona on for only a few samples.
- The share of questions with at least one "on" sample rises 0 → 0.06 → 0.25 → 0.62 over levels 3-5. The switch is
  probabilistic per sample and its rate grows with s.
- Per-item Qwen P(yes) is bimodal: 18% of items fall in [0.2, 0.8].
- Qwen under-calls the persona. At level 5, Claude's P(on) is 0.18 against Qwen's 0.087.
- Claude and Qwen agree on the on/off call for 97% of items. The Spearman is only 0.42 because almost all items are
  "off".
- Evil persona prompting reaches a 3.5× higher on-rate at a smaller KL. For evil, the steering vector is an
  inefficient route.

## W2/W3 (CONFIRMATORY replication of v1/v2): **FAILED to replicate** — `poc_v31_W2_W3_layers.png`

Endpoints: 1 pos-prompt and 1 neg-prompt sample per (question, paraphrase), 60 fresh questions per trait. The
direction is the mean response-token residual, pos minus neg.

- **W2.** Within-question split-half cosine (the noise ceiling) rises with depth, 0.32 → 0.53 for sycophancy and
  0.37 → 0.60 for optimism. Across-question cosine rises in parallel, 0.20 → 0.29 and 0.21 → 0.36.
  - The ratio across/within is flat at 0.51-0.61 at every layer.
  - The pre-registered prediction "ratio lower at layers ≥17 than ≤14" fails for optimism (0.59 vs 0.57). For
    sycophancy the boolean is satisfied only by rounding (deep mean 0.550 vs mid 0.548; not a meaningful difference).
  - **Verdict: not replicated.** v1/v2 found a ratio of about 0.6-0.7 at L21 against about 0.9 at L14, using
    last-prompt-token activations with adverb prompts. With response-token means and fresh own-trait questions, that
    contrast disappears.
- **W3.** Steering the neg-prompted model toward the pos-prompted one (teacher-forced per-token KL; 30 train and 30
  test questions):
  - Cluster directions never beat the global direction. The gap (KL_global − KL_cluster)/KL_global lies between
    −0.041 and +0.004 at all layers. Its CI excludes 0 on the negative side at sycophancy L14 and L17, and at optimism
    L3, L17 and L20.
  - **Power caveat.** Some KMeans clusters hold only 1-2 of the 30 train questions (sizes such as [2, 9, 9, 10] and
    [9, 13, 1, 7]). There, the ratio of estimator noise to signal (tr Cov / n / |v_glob|²) is about 0.4-1.6, against
    about 0.1-0.2 for an average cluster. Noise inflates the direction's norm and raises the KL.
  - So W3 is a **null with uncontrolled estimator noise**, not a demonstration that context dependence is absent. A
    fair rerun would use norm-matched or shrinkage cluster directions and a minimum cluster size.
  - The per-question cross-fitted "oracle" is worse than global at every layer (for example sycophancy L14: 0.50 vs
    0.33). It is built from only 4 single-sample differences, so estimator noise dominates any question specificity.
  - The pre-registered boolean "gap deep > mid" is true for sycophancy and false for optimism. In both traits the gap
    is ≤ 0 everywhere, so **the substantive prediction (cluster gain grows with depth) is not supported**.
  - Global steering is most effective at L14: KL drops from 0.54 to 0.33 for sycophancy and from 0.59 to 0.41 for
    optimism.
- **Interpretation.** On own-trait questions with all-position steering, a single global direction is as good as
  IDEEA-like cluster directions at every depth for these two traits. Any context dependence that exists is smaller
  than the estimation noise of a cluster mean (about 7 questions × 5 samples). The v1/v2 "context dependence grows
  with depth" result should be treated as setup-specific (last-token, adverb prompts, mixed-trait questions) until it
  is shown otherwise.

## A1 (EXPLORATORY): sycophancy × optimism grid — `poc_v31_A1_grid.png`

24 questions that carry both an opinion and an outlook (12 with a pessimistic user stance, 12 optimistic), 4×4 grid in
(s_syc, s_opt) ∈ {0, 1/3, 2/3, 1}·s_max, 20 samples per cell.

- **The interaction is negative everywhere and mostly a ceiling artefact.** On optimistic-stance questions both scores
  start at 0.68 and reach 0.96-1.0 after one step on either axis. The additive prediction then exceeds 1, so
  "interaction" equals −(overshoot), −0.12 to −0.29 (CIs exclude 0).
- **Pessimistic stance, sycophancy score.** The interaction reaches −0.54 at (1, 1), CI [−0.72, −0.37]. The two
  vectors each raise the sycophancy score, from 0.05 to 0.63 (syc) and to 0.91 (opt), but they saturate jointly.
  This is again consistent with a bounded score more than with a mechanistic interaction.
- **A cross-effect the design did not anticipate.** On pessimistic-stance questions, the optimism vector raises the
  sycophancy score strongly (0.05 → 0.91). The sycophancy vector does NOT make the answers more pessimistic: the
  optimism score goes 0.07 → 0.22.
  - The blind Claude labels on grid items agree in direction, but n is tiny: 3 items at (0, 0) and 6 at (0, max). On
    pessimistic-stance questions at (0, s_opt max), Claude rates sycophancy 3.75 against 1.33 at (0, 0).
  - **Raw texts explain it** (reviewer check of `gen_grid_03`). The optimism-steered model opens with "I completely
    agree with you!" and then argues the opposite. For example, "oceans mostly dead" is answered with "coral reefs
    coming back".
  - **This is a judge-construct problem**: surface agreement is scored as sycophancy. It is not a real optimism→sycophancy
    spill-over. A sycophancy judge for this design must separate stated agreement from the substance of the stance.
- **Bounded scores make additive diff-in-diff uninformative near the ceiling.** A link-scale analysis (logit of the
  rate) would be needed. This is not pre-registered and was not done.

## A2 (CONFIRMATORY test): first-order spill-over prediction — `poc_v31_A2_spillover.png`

- **Predictor.** The predicted slope is dT_B/dc = Cov(s_B, d log p(y|c u_A)/dc) from 40 independent unsteered
  samples per question (autograd). Sanity check: the mean of the score over samples is about 0 (0.00-0.02, sd 0.4).
- **Observed.** The observed slope is the first grid step.
- **Pre-registered test passes.** Spearman(predicted, observed spill-over) across 24 questions × 2 directions =
  **0.36 [0.10, 0.61]** (bootstrap over questions).
  - Per direction: opt→syc 0.46; syc→opt 0.16 (n = 24 each, not separately powered).
  - Own-trait calibration check: ρ = 0.42 [0.10, 0.69].
  - Predicted magnitudes are smaller than observed: the regression slope of observed on predicted is 0.17-0.36, and
    for opt→syc mean predicted 0.010 vs observed 0.019 per unit c. The first step (s = s_max/3, i.e. 0.4-0.5) is not
    infinitesimal.
- **But a simple baseline predicts better.** The baseline is "Euclidean cosine × that question's observed own
  steerability of B". It uses steered data; ours uses only unsteered samples. With disjoint sample halves, so that no
  grid noise is shared, it reaches ρ = 0.61 [0.41, 0.77] against ours 0.37 [0.11, 0.61] (paired difference CI
  [−0.50, 0.02]). The covariance-limit baseline reaches 0.29 [−0.06, 0.61].
- **Pre-registered baselines (they share grid noise with the observed value):** Euclidean ρ = 0.79, covariance limit
  ρ = 0.47. Both are above ours.
- **Trivial headroom baseline.** Using only 1 − T_B(0,0) from the base samples gives ρ = 0.34, against ours at 0.36
  (difference CI [−0.44, 0.37]). Ours is uncorrelated with headroom (ρ 0.03), so they carry different information.
- **Pooling inflates our ρ slightly.** Part of the pooled 0.36 comes from the two directions having different means.
  Rank-pooled within direction it is 0.31 [0.003, 0.59].
- **Conclusion.** The score-function first-order prediction carries real per-question signal from base samples alone.
  However, most of the predictable variation in spill-over here is "how steerable is trait B on this question", which
  steered data capture better. This does not yet support H2 ("metric K_ij predicts spill-over better than the
  baselines").

## A3 (EXPLORATORY): first-order least-interference correction within span{u_A, u_B} — `poc_v31_A3_correction.png`

β is fitted on 8 DEV questions as a ratio of mean covariances.

- **Steer sycophancy, hold optimism.**
  - Global β = −0.15, so the correction is negligible. Spill-over onto optimism is +0.22 before and +0.25 after (at
    level 3, chord s ≈ 1.21).
  - The stance-specific β is 11.3 for pessimistic stance. The ratio estimator blew up because its denominator is near
    0 with 4 questions. The "corrected" direction is then mostly −u_opt, and it REVERSES sycophancy (dA = −0.25).
  - **Failure.**
- **Steer optimism, hold sycophancy.**
  - Global β = 2.36. Spill-over onto sycophancy goes from +0.59 to −0.29, an overshoot.
  - Own-trait gain falls from +0.28 to +0.07, at a lower realised chord (1.04 vs 1.44).
  - The stance-specific β gives a similar picture.
  - **Failure.** First-order correction within a 2-d span with β from 8 questions overcorrects and destroys most of
    the intended effect.
- **Diagnosis from TEST data (explanatory only, not used to refit).** The observed first-step ratios imply β ≈ 0.86
  (steer optimism, hold sycophancy; fitted 2.36) and β ≈ 0.78 (steer sycophancy, hold optimism; fitted −0.15, the
  wrong sign). So the covariance predictor itself is off on the 8 DEV questions; a small denominator is not the only
  problem.
- With β = 2.36, the normalised d keeps only 0.46 weight on u_opt, so the loss of own-trait gain is built in.
- The comparison is NOT at matched realised s, as the design required: chord 0.74 vs 1.10, and 1.04 vs 1.44.
- A3 therefore shows only that this first-order correction, with β from 8 questions, fails. It says nothing either way
  about least-interference flows in general.

## What is solid vs exploratory

- **Solid (confirmatory, pre-registered, both vector sets; Claude validates the judge and the endpoints):**
  - For sycophancy and optimism, realised per-token KL arc length along a norm-normalised persona vector passes the
    pre-registered criteria.
  - The mean trait score, a rate of near-binary verdicts, rises with s with a trend of the expected size. Its shape
    (near-linear for sycophancy, concave for optimism) rests on Qwen level means only.
  - The coordinate transfers to fresh questions with steps even to ±7%.
  - The thresholded Qwen judge is valid (ρ 0.75-0.85 vs Claude).
- **Negative (pre-registered replication failed):**
  - W2 does not replicate the v1/v2 claim that context dependence grows with depth.
  - In W3, cluster steering never beats global. W3 is a null with uncontrolled cluster-estimator noise, not proof that
    context dependence is absent.
- **Confirmatory but weak:** the A2 first-order spill-over predictor correlates with observed spill-over (0.36), but
  it is not better than the pre-registered baselines or a trivial headroom baseline.
- **Exploratory:** KL growth along the path, per-cluster slopes, the evil on-rate, A1 (ceiling-dominated interaction,
  plus a judge-construct artefact: the optimism vector produces "I agree" openers that the judges count as
  sycophancy), A3 (correction fails; β is wrong in sign or size).

## Deviations and caveats

1. analyze.py was patched after the runs for crashes only: a JSON int/bool cast, KMeans with fewer questions than
   clusters (smoke only), and an SVD guard. No criterion or statistic changed.
2. The "two paraphrase sets" were interpreted as two disjoint instruction-paraphrase sets for the vector. Questions
   were not paraphrased. This is stated in the design.
3. The Claude K1 check pools sets A and B (levels 1-5 are split across sets in the blind subset). With 22 items per
   level, Claude's level means are noisy. Sycophancy level 3 dips to 2.93, against 3.09 at level 0. Claude does not
   independently confirm the shape of T(s), only the endpoint effect.
4. A2 baselines as pre-registered share grid noise with the observed spill-over. The split-half re-computation in
   `a2_split_half.json` is post hoc and is reported alongside.
5. Labellers were Sonnet subagents reading full (≤128-token) answers. Each pass took under a minute and wrote labels as
   ordered arrays. Optimism pass 3 (explicit id→label pairs) was added after review; see K0 above. A future protocol
   should always require id→label pairs.
6. K6 uses the mean of the two sets' sd_pool, a detail the design did not specify.

## Review
An adversarial Opus reviewer read the code, the criteria and the conclusions, and recomputed from the raw files.

**No code bug was found.** The arc length, K1/K4-K6, the Claude-governs logic and the A2 autograd all match the design.

The following changes were made after review:
- the chord ≈ arc claim is restricted to sycophancy;
- the limits of K2/K3 are stated, and T(s) is called a population rate;
- W3 is downgraded to a null with uncontrolled estimator noise;
- the pre-registered A2 baselines and the headroom baseline are added;
- the A3 "estimator, not idea" claim is removed, and the unmatched-s caveat added;
- the A1 cross-effect is reattributed to "I agree"-then-contradict texts, a judge-construct issue;
- the labels were checked for alignment and an optimism pass 3 added;
- the length-confound and sequence-arc numbers are added.

## Figures
| figure | content | status |
|---|---|---|
| `poc_v31_W1_trait_vs_s.png` | T(s) per trait and set, per-question curves coloured by cluster, oracle line, Claude levels | CONFIRMATORY |
| `poc_v31_W1_evil_on_off.png` | evil P(persona on) vs s | exploratory |
| `poc_v31_W2_W3_layers.png` | W2 cross-q vs within-q cosine by layer; W3 cluster/oracle KL gain vs global by layer | CONFIRMATORY replication, failed |
| `poc_v31_A1_grid.png` | observed, additive and interaction grids by user stance | exploratory |
| `poc_v31_A2_spillover.png` | predicted vs observed spill-over; own-trait check | CONFIRMATORY, weak |
| `poc_v31_A3_correction.png` | spill-over before and after the first-order correction | exploratory, failed |

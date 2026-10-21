# OT geometry of unpaired POS/NEG head activations — preregistration

Written 2026-10-20, **before** any OT statistic was computed. (Only the data shape was looked at:
row counts per class.) Papers: W ("Is steering space flat?") and C (support boundary).

## Data
- Gemma-2-2B-it, TruthfulQA, IDEEA harness cache `~/scratch/ideea_gemma/results/truthfulqa/gemma2_2b/activations/head.pkl`
  (o_proj input, last token, one row per reference answer; label +1 correct / -1 incorrect), restricted to
  the **development** split (train+val questions) exactly as `IDEEAFit.fit` does. Test questions are never loaded.
- Question split for anything predictive: IDEEA's own `train` questions = **fit set**; `val` questions = **held-out set**.
- Heads: the top-24 heads of IDEEA's ranking (`sorted_idx` from ITI probe val accuracy; paper config K=24),
  plus 8 random control heads (numpy seed 0, drawn from the remaining 184).

## Spaces (standardisation)
- **Primary: z** = per-coordinate z-score with mean/std of the pooled (POS∪NEG) fit set of that head.
- Sensitivity: **raw** (no transform), **pw32** (PCA to 32 dims on the pooled fit set, whitened).
- Displacement vectors transform linearly between spaces; IDEEA's KMeans is always fitted in raw space
  (as in its code) and its direction is then mapped into the analysis space.

## (a) Entropic OT
- Uniform marginals on NEG (n−) and POS (n+) fit samples (class imbalance handled by the marginals), cost
  C_ij = ||x_i − y_j||², ε = c · median(C), c ∈ {0.01, 0.03, 0.1, 0.3}; log-domain Sinkhorn (own torch code,
  tol 1e-4 on marginals, ≤ 3000 iters). Barycentric map T(x_i) = Σ_j P_ij y_j / Σ_j P_ij, d_i = T(x_i) − x_i.
- Primary ε: c = 0.03. Others reported.

## (b) Flatness
- **R²_flat** = ||mean d||² / mean ||d_i||² (share of the displacement field's second moment explained by a constant).
- **cos̄** = mean_i cos(d_i, μ+ − μ−).
- Nulls (same n−, n+, same space, same ε, same pipeline):
  1. **G-shared**: two Gaussians with the fitted means and the pooled within-class covariance (truth = translation).
  2. **R-shift**: class-centred residuals pooled and randomly reassigned to the two classes, then the class means
     added back (truth = translation, but keeps the data's non-Gaussian / multimodal shape).
  3. **G-sep**: Gaussians with separate class covariances (truth = affine, non-constant map).
  4. **Perm**: label permutation (truth ≈ identity; checks that any coherent displacement exists).
  Reps: 20 for G-shared and R-shift, 10 for G-sep and Perm. Null 95% interval = 2.5–97.5% quantiles of the reps.
- **Verdict rule (primary ε, z space, top-24 heads)**:
  - head is *flat* if R²_flat ≥ the 2.5% quantile of BOTH translation nulls (G-shared, R-shift);
  - head is *curved* if R²_flat < the 2.5% quantile of both translation nulls;
  - head is *affine-explained* if curved w.r.t. translation nulls but ≥ the 2.5% quantile of G-sep.
  - **"Flat" overall** if ≥ 70% of top-24 heads are flat. **"Curved" overall** if ≥ 50% are curved
    (X = 50%). Otherwise "mixed". The result must hold at ≥ 3 of the 4 ε values to be called robust.
- Prediction (written now): **curved overall** with respect to G-shared, but a large part of that is
  explained by R-shift (multimodality from question clustering) — i.e. I predict the R-shift null is much
  lower than G-shared and the data sit closer to R-shift. Second prediction: random control heads look
  *flatter* relative to their nulls than top heads? No strong prior; recorded as exploratory.

## (c) Geodesic vs Euclidean
- kNN graph (k = 10, symmetrised, Euclidean edge weights) on POS∪NEG fit points; Dijkstra shortest paths.
- For each NEG point i, endpoint j*(i) = argmax_j P_ij (a POS node). Ratio ρ_i = geo(i, j*) / ||x_i − y_j*||.
  Statistic = median ρ (disconnected pairs counted and excluded). Compare with the same statistic under
  G-shared and R-shift (10 reps each, primary ε).
- *Curved-geodesic* if median ρ > the 97.5% quantile of both translation nulls.

## (d) Does IDEEA's clustering capture the displacement field?
- Target: d_i from OT on the full development set (fit+held-out NEG vs fit+held-out POS), primary ε.
- Predictors (all fitted on fit questions only), evaluated on held-out NEG points by cos(pred, d_i):
  1. IDEEA `nearest_pos_neg`, nc = 2 (paper nc), KMeans seed 0, n_init 10, raw space; also nc = 4.
  2. IDEEA `min_perp`, nc = 2 (the paper's selected variant), for reference.
  3. Global mean difference μ+ − μ− (fit set).
  4. kNN-local average of the full-OT d over the k = 20 nearest fit-NEG points (k fixed now, not tuned).
- Ceiling reference: the same predictors evaluated under the R-shift null (where truth is constant).
- Prediction: kNN-local > mean-diff ≥ IDEEA npn by a margin of ≥ 0.05 in mean cos on ≥ 50% of top heads;
  if kNN-local does not beat mean-diff by ≥ 0.05, "local structure of the field is not predictable" (supports flat).
  IDEEA beats mean-diff by ≥ 0.05 on ≥ 50% of top heads ⇒ "clustering captures curvature".

## (e) Boundary
- Held-out NEG x moved along x + t·d for t ∈ [0, 3] (step 0.1), with d = its OT displacement (target from (d))
  and, separately, d = μ+ − μ− (CAA/ITI-like constant step).
- In-distribution band: distance from held-out POS points to their k-th (k = 10) nearest fit-POS point;
  band = ≤ 95% quantile q95.
- Per point, t_exit = the largest grid t at which the point is still inside the band (NaN if never inside).
  Report median t_exit and the fraction inside the band vs t.
- Prediction: median t_exit ∈ [1, 2] (points leave POS support soon after the nominal full transport t = 1).
- α ↔ t: IDEEA adds α·naive_std·v̂ per head. Equivalent t along mean-diff = α·naive_std / ||μ+ − μ−||_raw
  (projected). Relate α = 4 (paper config, TI .572) and α = 15 (CLI default, TI .099 < base) to t_exit.
  Prediction: α = 4 lies inside the band, α = 15 outside. No GPU run.

## Not done / caveats known in advance
- Rows of the same question share the question context; POS and NEG of a question are therefore *de facto*
  paired, which IDEEA ignores. The R-shift null keeps the class-common shape but breaks per-question imbalance.
- 256-d with ~1–2k points per class: OT estimates are dominated by sampling noise; nulls are the point.

## Amendment A1 (2026-10-20, still before any real-data OT statistic; extraction job had not even started)
A synthetic power check (scratch script; 2-mode NEG, n−=900, n+=600, field = two orthogonal shifts of size 3,
true field R² = 0.5, vs a flat control with the same ||Δμ||) showed that the raw **R²_flat has no power in 256-d**:
the finite-sample matching noise (≈ 2·dim) swamps the field (flat 0.034 / null 0.037; curved 0.017 / null 0.020).
A kNN-smoothed field does have power:
| dim | stat | flat data / null min | curved data / null min |
|---|---|---|---|
| 8 | R²_s50 | 0.985 / 0.985 | 0.531 / 0.953 |
| 32 | R²_s50 | 0.910 / 0.907 | 0.486 / 0.830 |
| 256 | R²_s50 | 0.613 / 0.632 | 0.348 / 0.484 |
(R-shift null, 5 reps, c = 0.01.) Changes, all fixed now:
1. **Primary flatness statistic = R²_s50**: d̃_i = mean of d over the 50 nearest NEG neighbours of x_i (self excluded),
   R²_s50 = ||mean d̃||² / mean ||d̃_i||². The verdict rule of (b) is applied to R²_s50 instead of R²_flat;
   R²_flat and cos̄ (also cos̄ of d̃) are reported as secondary.
2. **ε grid** c ∈ {0.003, 0.01, 0.03, 0.1}; **primary c = 0.01** (at 256-d, c = 0.03 already gives ~70 effective
   partners per point and c = 0.3 a near-uniform plan, so the old grid was mostly in the blurred regime).
3. Added sensitivity space **pw8** (PCA-whitened 8 dims), where OT is well estimated.
4. The "flat" verdict on real data is only interpretable together with the power check above; the report
   must state the minimum detectable curvature at the real n and dim (re-run the synthetic at matched n).

## Amendment A2 — REFRAME (2026-10-20, PI via coordinator): the trait flow is the primary object; OT is a baseline column
Written before any trait-flow code was run. OT analyses (a)–(e) above continue unchanged as a comparison column
(array job 6593005 was already queued; none of its outputs had been read when this was written).

### Objects
- **Trait score** s(x) ≈ log p+(x)/p−(x) (up to a constant), fitted per head on UNPAIRED fit-set rows, z space:
  (i) **LR**: L2 logistic regression, C by grouped (by question) 3-fold CV over {0.001, 0.01, 0.1, 1};
  (ii) **MLP**: one hidden layer, 64 tanh units (smooth), L2 weight decay α by grouped 3-fold CV over {0.01, 0.1, 1, 10},
  lbfgs, max_iter 500, seed 0. ∇s computed analytically.
- **Trait flow** dx/dℓ = V(x), V = P_T(x)∇s / ||P_T(x)∇s|| (unit speed, ℓ = arc length);
  P_T(x) = projection on the top-r local PCs of the k = 64 nearest fit points (POS∪NEG) of x, r = smallest number of
  PCs explaining ≥ 90% of local variance. Also the **unprojected** flow V₀ = ∇s/||∇s||.
- Euler integration, step h = 0.02·||Δμ||, up to arc length 3·||Δμ|| (150 steps), from every held-out NEG point.
  (Δμ = fit POS mean − fit NEG mean, z space.)

### Limits stated in advance (checked in code)
- Linear s (LR) ⇒ V₀ constant (= probe weight w); this is the ITI/probe-like flat field. P_T(x)w is not constant, so the
  projected linear flow turns only through the tangent estimate.
- IDEEA nearest_pos_neg = hard-assignment, piecewise-constant estimate of a local displacement (not a gradient).
- OT displacement = minimum-kinetic-energy transport of the whole distribution; not the gradient flow of s in general.
- NOTE (stated now): a pure translation of a NON-Gaussian shape does NOT give a linear log-ratio, so the trait flow of
  a translated mixture is curved. Hence "OT-flat" ≠ "trait-flow-flat"; the equal-covariance Gaussian is the null in
  which both are flat.

### Analyses and decision rules
1. **Nonlinearity of s** (gate): MLP vs LR held-out log-loss and AUC on held-out questions, bootstrap over questions
   (1000). "Nonlinear s supported" if the MLP log-loss improvement has a 95% CI excluding 0. If not supported,
   the field is reported as flat-by-Occam regardless of 2.
2. **Flatness of flow lines** (MLP s; projected and unprojected): turning angle θ(ℓ) = ∠(V(x₀), V(x_ℓ)) at
   ℓ ∈ {0.5, 1, 2}·||Δμ||, and mean curvature κ̄ = mean_steps ||V_{k+1} − V_k|| / h. Per head statistic = median over
   held-out NEG starts. Nulls (same n's, same pipeline incl. the α selected on the data): **G-shared** (10 reps; the
   field must be constant), **Perm** (5 reps; no trait), **R-shift** (5 reps; data-shape translation — reported, not
   used for the verdict). Head is *curved* if θ(||Δμ||) > G-shared 97.5% quantile, *flat* if ≤ it.
   **Overall "curved"** if ≥ 50% of top-24 heads are curved (unprojected flow = primary, since the projected flow's
   turning includes tangent-estimation noise); **"flat"** if ≥ 70% are flat; else "mixed".
   Prediction: curved overall (the paired per-question δ_q check run earlier, post hoc, already showed a mostly
   non-constant per-question displacement), but with turning far below the Perm null.
3. **Estimator agreement** at held-out NEG points: cos(V(x), ·) for IDEEA npn nc=2, mean diff, OT displacement
   (full-dev entropic OT, c = 0.01, z space), LR weight w; both V and V₀. Also cos(V(x), mean_x V) (self-consistency).
   Prediction: cos(V₀, w) highest; IDEEA ≈ mean diff; OT lowest.
4. **Boundary along flow lines** (unprojected and projected MLP flow, and the straight line x + ℓ·Δμ/||Δμ||): first
   arc length at which (a) s saturates: ds/dℓ < 10% of its initial value; (b) the field vanishes: ||P_T∇s|| < 10% of
   initial; (c) the point leaves the POS support band (10-NN distance to fit POS > q95 of held-out POS), counted
   only after the path has been inside once. Report median arc lengths in units of ||Δμ||.
   **Prediction: the straight line leaves the support sooner** (smaller median ℓ_exit) than the projected flow, on
   ≥ 70% of top heads; for the unprojected flow no prediction (MLP extrapolation off-support is possible).
   Also report s at exit and the fraction never entering the band.
5. **Composition**: SKIPPED — no second label set is cached for Gemma (only TruthfulQA); a new trait would need a GPU
   extraction and a bridge experiment holds the GPUs. Noted as not done.
Heads: same top-24 + random-8. Space: z (primary), pw32 (sensitivity).

## Deviation log
- D1 (after A2, before any OT/flow output was read): because OT is demoted to a comparison column and CPU fairshare is low,
  the OT array was cut to spaces {z, pw8} (raw and pw32 dropped) with 10 reps for G-shared/R-shift (5 for G-sep/Perm)
  instead of 20/10. Trait-flow spaces: z (primary), pw32.
- D2 (after seeing the first trait-flow outputs; POST HOC, labelled as such in the report):
  (i) the A2 horizon 3·|Δμ| is too short — on every head finished so far no path (flow or straight line) leaves the POS
  kNN band and s never saturates, while IDEEA's own steps are t ≈ 5–20·|Δμ| (OT smoke). Boundary analysis 4 is re-run
  with a two-phase path: 150 steps of 0.02·|Δμ| (unchanged) then 0.25·|Δμ| steps to 30·|Δμ|; same exit/saturation/vanish
  definitions; IDEEA α ↔ arc length via t = α·naive_std/|Δμ_raw|. No nulls for the boundary (none were preregistered).
  (ii) grouped CV picked α = 10, the edge of the preregistered grid, on every head, and the z-space MLP had worse
  held-out log-loss than LR. Sensitivity: α grid extended to {10, 30, 100, 300} (z space), turning + G-shared null
  (5 reps) recomputed. The preregistered verdict uses the original run; the sensitivity is reported beside it.

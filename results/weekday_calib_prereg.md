# Prereg — weekday/month calibration of the coordinate-free geometry toolkit (paper W sanity check)

Written 2026-10-27, committed before any extraction or analysis.

## Why
`results/ot_geometry.md` concluded "steering space is flat" on TruthfulQA head activations. That toolkit was never shown to detect
curvature where curvature is known to exist. Wurgaft et al. 2026 report that weekdays and months lie on curved, cyclic
low-dimensional manifolds in LLM residual streams, and that steering along the manifold beats straight lines. We use that as a
**positive control**: a toolkit (or a statistic) that cannot see curvature here has no power, and its "flat" verdict elsewhere means nothing.

## Data (GPU job, one MIG slice)
- Models: Gemma-2-2B-it (primary); Qwen2.5-1.5B-Instruct (secondary, same code).
- Weekdays: z ∈ 7 days, k = 1..13, 6 templates ("What day is k days after z?", …) → 546 prompts.
  Months: z ∈ 12 months, k = 1..11, 6 templates → 792 prompts.
- Chat template, then forced assistant prefix "The answer is"; activation = residual stream (hidden_states[l], all l) at the last prompt token.
- Label = correct answer (day / month). Also stored: z, k, template, model top-1 among the class tokens, correctness.
  Primary analysis uses all points; a sensitivity analysis uses only correctly answered prompts.

## Toolkit (per layer, per model, per domain; CPU)
Space: centred, PCA-64 fitted on that layer's points (sensitivity: PCA-16, PCA-256 / full).
- **S1 TwoNN intrinsic dimension** (Facco et al.; 10% largest ratios discarded).
- **S2 Isomap geodesic** d_G on a symmetrised kNN graph (k = 10; grid k ∈ {5, 10, 20} only for tuning), Euclidean d_E in the same space.
- **S3 geodesic distortion** 1 − Spearman ρ(d_G, d_E) over all connected pairs.
- **S4 cyclic order recovery**: class-level distance matrices D_E (centroid distances) and D_G (mean d_G between members).
  Score = Spearman(D, circular index distance) − Spearman(D, linear index distance) ("cyclicity"); also ρ with circular distance alone.
  Caveat stated in advance: on a perfect circle the chord is monotone in arc length, so ρ_circ cannot separate d_G from d_E there;
  the separation can only come from noise / extra dimensions.
- **S5 local direction field**: for each point x of class i, v_i(x) = mean of its m = 10 nearest class-(i+1) points − mean of its m nearest
  class-i points. Statistics: (a) rotation of the mean field between successive steps i→i+1 and i+1→i+2 (degrees);
  (b) field coherence: mean cos(v_i(x), mean v_i).
- **S6 centroid curve**: closed polyline through the 7 (12) class centroids, turning angle θ_i between successive centroid differences
  (naive), and a **noise-corrected** turning angle from split halves: cos θ̂ = ⟨Δ_i^A, Δ_{i+1}^B⟩ / sqrt(⟨Δ_i^A, Δ_i^B⟩⟨Δ_{i+1}^A, Δ_{i+1}^B⟩)
  (symmetrised over A/B, averaged over 50 random splits). Also chord/arc ratio of centroids two and three steps apart
  (d_E(c_i, c_{i+2}) / (|Δ_i| + |Δ_{i+1}|)), and the split-half (cross-covariance) spectrum of the centroids: share of the top-1 and top-2 PCs.
  A straight concept direction gives top-1 share ≈ 1, corrected turning ≈ 0°, chord/arc ≈ 1; a regular 7-gon gives turning 51.4°
  (12-gon 30°), top-2 shares 0.5/0.5, chord/arc(2) = cos(θ/2).

Nulls (each statistic against each null where defined; 20 reps unless noted):
- **N-flat-G**: one Gaussian with the data's mean and covariance (labels kept by index) — for S1, S3, and as a "no structure" reference for all.
- **N-flat-line** (the null that "flat / linear concept" actually means): each class = (pooled within-class residuals, permuted)
  + a centroid placed on the best-fit straight line (centroids projected on their top PC). Point-level shape is real, the class arrangement is straight.
- **N-blob**: each class Gaussian with its own mean and the pooled within-class covariance (curved centroid arrangement, Gaussian blobs) —
  tells whether point-level statistics see more than the class means.
- **N-shuffle**: labels permuted (label-dependent statistics S4–S6).
- **N-resample**: within-class bootstrap of points (CIs for S6, S4).

Old statistics from `results/ot_geometry.md` / `results/klflow.md`, applied unchanged in spirit (binary class pairs (i, i+2) and (i, i+3),
NEG = class i, POS = class i+2 or i+3, z-standardised PCA-64, MLP α = 100, held-out half of templates as starts):
- **O1** flow-line turning after arc length |Δμ| (unprojected) vs the G-shared and R-shift nulls (10 reps each);
- **O2** frozen-Δμ path MMD diagnostics: MMD to the target class along the path (min location d*, overshoot), plus the new readout
  MMD to the **intermediate** class (i+1) along the straight path vs along the centroid polyline;
- **O3** the old geodesic/Euclid ratio `geo_ratio` (NEG→matched POS, k = 10).

## Predictions and pass/fail (layers "middle" = Gemma 6–18, Qwen 6–20; a statistic "detects" if it beats the named null in ≥ 2/3 of middle layers)
- P1 TwoNN ID of weekday/month points in middle layers ≤ 3, and below N-flat-G (expected ≫ 10) on ≥ 2/3 middle layers.
- P2 cyclicity(D_G) > 0 and > the 97.5% shuffle quantile; and cyclicity(D_G) ≥ cyclicity(D_E) (expected but may be weak given the chord caveat).
- P3 distortion > N-flat-G max and > N-flat-line max.
- P4 corrected centroid turning far from 0 (expected ≈ 51° weekdays / 30° months, ≥ 25° required) and above every N-flat-line rep;
  split-half top-1 share < 0.8.
- P5 (old statistics) — open question, no directional prediction: does O1 exceed the R-shift null; does O2 show the straight path missing the
  intermediate class relative to the polyline. We expect O1 to have low power (a binary log-ratio between two blobs is nearly straight even
  on a circle), which would mean the TruthfulQA "flat" verdict carries little information.

**Toolkit verdict.** PASS = P1, P3, P4 all hold for months in Gemma; then confirm on weekdays. If the toolkit FAILS on months,
tune k ∈ {5, 10, 20} and PCA dim ∈ {16, 64, 256} **on months only**, fix the best setting, then run weekdays once with it.
Each statistic is reported as "has power" / "no power" separately.

## Optional steering (same GPU job; ≤ 1 GPU-h)
Layers every 2 from 4 to 20 (Gemma) / 4 to 22 (Qwen). Centroids from templates 0–4; steering prompts = template 5 with answer Monday (January).
Add δ(t) at the last token of hidden_states[l]: straight δ = t(c_Wed − c_Mon); polyline via c_Tue; periodic cubic spline through all centroids.
Also Mon→Thu, Jan→Mar, Jan→Apr. t ∈ {0, 0.05, …, 1}.
Readouts: next-token P over the class tokens; **Tuesday bump** = max_t P(Tue) (and P(Tue) at t = 0.5); **teleportation** = max_t of the mass
outside the 7 (12) class tokens. Prediction (Wurgaft): spline/polyline gives a larger Tuesday bump than straight on most middle layers;
straight shows more off-class mass or skips. Pass if spline − straight bump > 0 on ≥ 2/3 of the steered middle layers (Gemma).

# KL gradient flow vs frozen velocity — preregistration (2026-10-21, written before any klflow code was run)

**Theory (agreed with the PI).**
- The Wasserstein gradient flow of KL(ρ‖p+) has particle velocity v_t = ∇log p+ − ∇log ρ_t.
- At t = 0 with ρ₀ = p−, this is v₀ = ∇s with s = log p+/p−. So ITI, CAA, IDEEA and our trait flow are the
  **frozen initial velocity**.
- OT (Benamou–Brenier) is a different, endpoint-fixed geodesic.
- For equal-covariance Gaussians, all of these coincide with Δμ.
- Hypothesis: frozen velocities overshoot past p+, while the full flow (or the guidance flow ∇s + γ∇log p+) stops once
  ρ_t ≈ p+.

**Data, splits and heads** are the same as `ot_geometry`:
- dev split; fit = IDEEA train questions; held-out = IDEEA val questions (79 q, 335 NEG and ~287 POS rows);
- heads: IDEEA top-24 + 8 random;
- space z is primary; pw32 is a data-only sensitivity check.

Known caveat from `ot_geometry.md`: the kNN support band mostly measures the radius of the data cloud.

## Estimators (fit questions only)
- **s:** tanh-MLP log-ratio with α = 100. This is fixed now; it is the interior CV optimum found in ot_geometry D2.
  LR (C by grouped CV) gives w.
- **∇log p+, two estimators:**
  - (K) Gaussian-KDE score on fit POS. Bandwidth h = c·(median 1-NN distance among fit POS),
    c ∈ {0.5, 0.75, 1, 1.5, 2, 3}, chosen by grouped (by question) 3-fold CV log-likelihood on fit POS.
  - (D) Denoising-score-matching MLP (256-512-512-256, SiLU), noise σ = h from (K), Adam 1e-3, 4000 steps × batch 256.
- **ρ̂_t (particle flow):** Gaussian KDE of the current particle cloud, with the same h.

## Paths from held-out NEG starts
All paths use Euler integration, 150 steps, with the median initial step = 0.1|Δμ|. Unit-speed paths therefore
reach 15|Δμ|.
1. straight along Δμ
2. straight along the probe direction w
3. frozen ∇s (unit speed, MLP s)
4. guidance v = ∇s + γ∇log p̂+, with the (K) and (D) scores, γ = k·med|∇s(x₀)|/med|∇log p̂+(x₀)|, k ∈ {0.25, 1, 4};
   fixed dt
5. particle KL flow v_t = ∇log p̂+ − ∇log ρ̂_t (KDE, same h); fixed dt

## Metrics (computed at every step)
- MMD² (unbiased, RBF) between the moved cloud and held-out POS. Bandwidth = median pairwise distance of
  (held-out NEG ∪ held-out POS) × {0.5, 1, 2}; ×1 is primary.
- Energy distance.
- Median s.
- Fraction of points inside the data band and the POS band (10-NN distance ≤ q95).
- x-axis: median particle displacement |x_t − x₀| / |Δμ|.
- Floor: MMD²(random 335 fit-POS rows, held-out POS).

Per path:
- minMMD = the minimum over steps; d* = the displacement at that minimum;
- rise = (MMD_final − minMMD) / (MMD_start − minMMD);
- **plateau** if rise ≤ 0.1; **overshoot** if rise ≥ 0.5.

## Predictions and decision rules (top-24 heads, z, primary bandwidth)
- **P2a (shape).** Paths 1–3 overshoot on ≥ 70% of heads, and guidance (K, k = 1) and the particle flow plateau on
  ≥ 70% of heads.
- **P2b (key).** The minimum of minMMD over {guidance K/D at all k; particle} is below minMMD(straight Δμ) on ≥ 70% of
  heads, AND the per-head gap exceeds the 97.5% quantile of the same gap under the nulls.
  - Nulls: R-shift and G-shared, 20 reps each.
  - For cost, the nulls use only the KDE methods: straight Δμ, frozen ∇s, guidance K with k = 1, particle.
  - Rationale: under the equal-covariance Gaussian null, Δμ is the exact transport, so any advantage there is an
    estimator artefact.
- **P1 gate.** P1 (GPU) runs only if P2a AND P2b hold. Otherwise P1 is not run, and the report says why.
- **Bandwidth sensitivity.** The P2a/P2b verdicts are recomputed at bandwidths ×0.5 and ×2, and are called robust only
  if all three bandwidths agree.

## P3: conditional OT within question
- For every held-out question with ≥ 1 POS and ≥ 1 NEG row, run exact uniform-marginal OT (linear program) between
  its NEG and POS rows. Each NEG row's displacement is its barycentric target − x.
- Statistic: mean cos(d, ∇s(x)) over held-out NEG rows, using the MLP s (α = 100) and also the LR w.
  It is compared with:
  - unconditional entropic OT (all dev NEG → all dev POS, c = 0.01, as before), using the same ∇s;
  - Δμ;
  - a shuffle null: each question's NEG rows matched to the POS rows of a random other held-out question (20 reps);
  - a split-half ceiling: questions with ≥ 2 POS and ≥ 2 NEG rows are split into halves A/B, and
    cos(mean d_A, mean d_B) is taken per question.
- **Prediction P3.** cos(conditional OT, ∇s) ≥ cos(unconditional OT, ∇s) + 0.10 on ≥ 70% of top heads, and above the
  97.5% quantile of the shuffle null.
- Caveat, stated now: there are 3–4 rows per side per question, so d is noisy; the split-half ceiling bounds the
  agreement that is attainable.

## P1 (GPU) — only if gated in
- Guidance steering in `ideea_harness_gemma`: for each steered head, add α·[ŝ-gradient + γ·score(x)] at the current
  activation.
- Arms vs IDEEA and ITI/Δμ on an α grid, compared at matched realised KL.
- Endpoint: strict blind full-answer labels; also hedge / "I have no comment" and gibberish rates vs α.
- Constraints: ≤ 1 GPU job at a time, MIG slice, small batch, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- A prereg addendum is written before launch.

## Amendment A1 (before the main run; only a 5-step smoke test on L13H5 seen)
The KDE bandwidth CV picked c = 0.5, the lower edge of the grid, on the smoke head. The grid is extended to
c ∈ {0.25, 0.35, 0.5, 0.75, 1, 1.5, 2, 3}. If the CV still picks the edge, that edge is used and the report says so.
(Cause: in 256-d, KDE likelihood CV favours small h, which makes the KDE score close to "snap to the nearest POS row".)

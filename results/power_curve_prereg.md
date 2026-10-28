# Prereg — planted-curvature power curves for the geometry statistics (paper W)

Written 2026-10-28, committed before any planted simulation is run. Code: `power/`. Raw outputs: `~/scratch/power/`.
Compute: CPU only. Narval is in the POWEROUT20261027 reservation until 2026-11-03, and that reservation covers **all** CPU and GPU
compute nodes (`sinfo`: 0 idle nodes in every partition). Everything therefore runs on the login node, with single-thread BLAS
per worker (DEVLOG 10-27: multithreaded BLAS inside a Pool stalls).

## Why
- `results/weekday_calib.md`: the toolkit's power depends on SNR. Class statistics succeed at the weekday z token (class SNR ≈ 1.8)
  and fail at the last token (≈ 0.01).
- TruthfulQA top-24 heads have class SNR .004–.018. The `results/ot_geometry.md` "flat" verdict therefore has **no demonstrated power**.
- What is missing: a power curve on **real** activation clouds, with known planted structure. With it, every future "flat" or "curved"
  claim can carry a power statement, and we can size the persona-trait study.

## Backgrounds (real clouds; any real label signal is removed first)
Each background is a matrix of real rows, plus a group ("question") id. The real class means are subtracted, so no real ordered
structure remains:
| id | source | space | rows | groups |
|---|---|---|---|---|
| `tqa` | IDEEA `head.pkl` dev split, heads L13H5 (primary), L21H6, L15H2 | per-coordinate z (ot_geometry space), 256-d | 3049 | 395 questions (POS/NEG means removed) |
| `wk` | `~/scratch/weekday/gemma_weekday.npz`, last token, layer 12 | raw residual, 2304-d | 546 | 78 (k, template) groups (z-class means removed) |
| `wkz` | `gemmaztok_weekday.npz`, z token, layer 12 | raw, 2304-d | 546 (dedup 378) | (k, template) |
| `syc`, `opt` | PoC v3.1 `w23_acts.npz` (pos and neg prompted-response mean residual), layer 14 | raw, 2048-d | 600 | 60 questions (pos/neg means removed) |

Population within variance **W = trace of the background covariance** (all rows). It is reported with the participation ratio
(effective dimension) and the share of W that is between-question.

## Planting
L ordered levels, t_ℓ = ℓ/(L−1), L ∈ {2, 5, 7}. A row at level ℓ is: background row + a·γ(t_ℓ, x), with γ in a 2-D plane P.
- **P (plane)**: a random orthonormal 2-plane inside the background's row span (primary), or the background's top-2 PC plane (sensitivity).
- **Shapes γ**:
  - `arc θ`: unit-speed circular arc with total turning θ ∈ {0, 30, 60, 90, 180, 360}°, plus 308.6° (= 7 equally spaced points on 6/7
    of a circle, the weekday-like case). θ = 0 is the straight line, i.e. the **null**.
  - `sat`: a straight line with saturating spacing f(t) = (1 − e^{−3t})/(1 − e^{−3}). This is a second null: straight, unequal steps.
    It is the trivial way a graded trait can look "nonlinear in level" without any bending.
  - `bend θ`: two straight segments, the break at t = .5, θ ∈ {60, 90}.
  - `clust θ` (IDEEA-like): background rows are split into 2 latent clusters (KMeans on the background, fixed). A row of cluster k is
    shifted by t_ℓ·a·u_k, with u_0, u_1 in P at angle θ ∈ {60, 90, 180}.
    The level centroids then lie on a straight line by construction, so centroid statistics cannot see it.
- **Scale**: a is set so that the class SNR = (1/L)·Σ_ℓ ‖ā_ℓ − ā‖² / W equals the target. This is the same definition as
  `weekday/fourier.py` and `tqa_snr.py`. For `clust`, a is the straight-line amplitude at that SNR, so the per-point shift energy is matched;
  the achieved centroid SNR is reported as well. Target SNR ∈ {1e-3, 3e-3, .01, .03, .1, .3, 1, 3}.
- **Sampling**: n rows per level, n ∈ {20, 50, 100, 300, 1000}. Background rows are drawn without replacement while the background has
  enough rows, otherwise with replacement; this is flagged as a limitation for point-level statistics.
  - `iid` mode: rows drawn independently.
  - `q` mode (questions × levels): Q = n/5 questions, r = 5 replicates per (question, level).
    A row = μ_q + ε, where μ_q is a real question mean and ε a real within-question residual.
    - `q-raw`: no demeaning.
    - `q-dm`: the per-question mean (over all its levels and replicates) is subtracted.
    - SNR is defined with W of the raw background (it includes question variance), as for the real-data SNR numbers.
    - Stated in advance: in a balanced design, μ_q cancels exactly in centred level centroids. So centroid statistics are identical
      under `q-raw` and `q-dm`, and equal `iid` on the within-question residual cloud. Only point-level statistics can differ.
      This is checked numerically on one cell.
- Common random numbers: within one simulation, the same background draw is used for every shape and SNR.

## Statistics (all oriented so that "larger = more curved / more structure")
Centroid level. A/B = random within-level split halves (q mode: replicates split within question); 10 splits per simulation.
- **C1 `turn`** (weekday kit S6, open curve): split-half corrected turning angle at the L−2 interior vertices, median.
- **C2 `offline`** = 1 − top-1 share of the split-half cross-Gram of the centred centroids (weekday "top-1 share").
- **C3 `quad⊥`** (new): orthonormal polynomial contrasts p1, p2 over t_ℓ. b1 = Σ p1_ℓ c_ℓ (pooled data), û = b1/‖b1‖.
  Statistic = ⟨P⊥ b2^A, P⊥ b2^B⟩ / ⟨b1^A, b1^B⟩, with P⊥ = I − ûûᵀ: the reliable quadratic-in-level power **orthogonal to the line direction**.
  - The projection is there so that `sat` (a straight line, nonlinear in level) is not called curved.
  - Its permutation test is a **residual permutation under the fitted straight line**.
  - A level-label permutation would test order, not curvature, so it is not used.
- **C4 `ord1`** (order, not curvature): share of the split-half Gram along the linear contrast p1, against an **exact order-relabel
  null** (all L! permutations for L = 5, 5040 for L = 7). Reported with the weekday-kit circular f1 share (`f1c`, same null).
  This is a test of **graded ordered structure** (a straight line passes it). Its null is SNR = 0 (no structure).

Point level, PCA-64 of the pooled planted sample (weekday kit default):
- **P1 `frot`**: local direction-field rotation (kit S5, open): median angle between successive mean local fields ℓ→ℓ+1.
- **P2 `fincoh`** = 1 − local-field coherence (mean cos of v(x) with the level-pair mean field, averaged over pairs). Defined for L = 2.
- **P3 `dist`**: geodesic distortion 1 − Spearman(d_G, d_E), kNN k = 10. Only for n·L ≤ 2100 (all-pairs Dijkstra cost).

Binary, level 0 vs level L−1 (the ot_geometry formulation):
- **B1 `O1`**: flow-line turning after arc length |Δμ|, MLP α = 100, unprojected, fitted on a fit half, started from held-out level-0 rows
  (`ot_geometry/trait_flow.py` code, as used by `weekday/old_stats.py`).
- **B2 `OTr2`**: 1 − OT R²_s50 (Sinkhorn ε = .01·median cost, kNN-smoothed displacement field, k = min(50, n_fit − 1)).
- Space: `tqa` in its 256-d z space (as in ot_geometry); other backgrounds PCA-64 then z (as old_stats).

## Calibration and power
- **Oracle power** (main grid): the α = .05 threshold of each curvature statistic is the 95% quantile of that statistic over the
  `arc 0` simulations at the same (background, plane, L, n, mode, SNR). Power = P(statistic > threshold) for the other shapes.
  This is the best power any calibration of that statistic can reach.
- **Practical tests** (check subset): C1–C3 and P1–P3 against the **line null** (centroids projected on the top PC of the observed
  centroids; residuals permuted across levels, within question in q mode; 99 reps). B1–B2 against the **R-shift null** (10 reps,
  as in ot_geometry). Measured: FPR on `arc 0` and `sat`, and power on `arc 90`.
  - If a practical test has FPR ≤ .075 and its power is within .1 of oracle power, the oracle maps stand in for it.
  - Otherwise both are reported and the practical one governs any claim.
  - Check subset: backgrounds tqa and syc; L = 7; n ∈ {50, 300}; SNR ∈ {.03, .3, 3}; 100 simulations.
- C4 uses its exact order-relabel p-value everywhere (no oracle needed). Its FPR is measured at SNR = 0.
- **Simulations per cell**:
  - centroid grid: 400;
  - point grid: 60;
  - binary grid: 40.
  The binomial SE at power .8 is .02 / .05 / .06 respectively.

### Grids (sized for a shared login node; any cut is reported)
- Centroid grid: all 6 backgrounds (tqa L13H5 plus 2 extra heads); planes random / top-PC; L ∈ {5, 7}; all n, all SNR, all shapes;
  modes iid / q.
- Point grid:
  - backgrounds tqa, wk, syc; random plane; L ∈ {2, 5, 7}; n ∈ {20, 50, 100, 300};
  - SNR ∈ {.01, .03, .1, .3, 1, 3};
  - shapes arc 0/60/90/180, bend 90, clust 90/180, sat;
  - modes iid / q-raw / q-dm (q modes only for tqa and syc).
- Binary grid:
  - backgrounds tqa (L13H5), syc; L = 2; n ∈ {100, 300, 1000, 1200};
  - SNR ∈ {.003, .01, .03, .1, .3, 1};
  - shapes arc 0, clust 60/90/180; iid.
  - For L = 2, arcs and bends are **unidentifiable by construction** (two points define a line). Their power must equal the FPR; this is a sanity check, not a result.

## Real-data placement (measured with the same SNR definition)
- TruthfulQA top-24 heads: binary, L = 2, n ≈ 1200 per class (fit ≈ 2.4k rows).
- Weekday: last token (z labels, L = 7, n ≈ 78) and z token (n ≈ 54).
- PoC v3.1: pos vs neg at the 7 cached layers (L = 2, n = 300). The W1 design (30 TEST questions × 6 levels × 20 samples) is placed as
  L = 6, n = 600, at the SNR implied by pos/neg. A linear ladder has SNR_L = SNR_2·4·var(t). For L = 6 that is ×.47.

## Predictions and pass/fail (fixed now)
- **P1 (validity).** Every curvature statistic's practical test has FPR ≤ .075 on `arc 0`. C3 `quad⊥`, C2 and C1 also have FPR ≤ .075 on `sat`.
  A statistic that fails is declared invalid for claims.
- **P2 (weekday consistency).** On the `wk` background, `arc 308.6`, L = 7, n = 78:
  - at SNR .011, oracle power of C1 `turn` < .5;
  - at SNR 1.8 (`wkz` background, n = 54), power > .9.
  This reproduces the real weekday outcome. If it fails, the benchmark does not track reality and the decisions below are not drawn.
- **P3 (scaling).** For centroid statistics at θ = 90, L = 7, n ≥ 50, the 80%-power contour satisfies n·SNR ≈ const within a factor of 2.
- **P4 (blindness).** Centroid statistics C1–C3 have power ≤ .10 on `clust` plants at every cell. At least one of P2 `fincoh`, B1, B2 has
  power ≥ .8 on `clust 90` somewhere in its grid.
- **P5.** C4 `ord1` reaches 80% power on `arc 0` at a lower SNR than any curvature statistic reaches 80% on `arc 90` (same n, L).
- **P6.** C3 `quad⊥` is at least as powerful as C1 `turn`: its 80% SNR at θ = 60 and θ = 90 is ≤ that of C1, at L = 7, for n ≥ 50.

## Decision rules
- **TruthfulQA verdict.** Look at the IDEEA-like `clust 90` power of B1 O1 at n = 1200 and SNR .01 on `tqa`.
  - If it is < .8, the ot_geometry "flat" verdict is renamed **"underpowered (no power)"**.
  - If it is ≥ .8, the verdict is restated as "no IDEEA-like heterogeneity of angle ≥ 90° at SNR ≥ .01".
  - In either case, curved trajectories through intermediate intensities are unidentifiable from a binary label (L = 2).
- **Statistics for the persona study.** Choose the 1–2 statistics that pass P1 and have the lowest 80%-power n at θ = 60 and θ = 90
  (L = 5 or 7, at the measured persona SNR). One must be a curvature statistic. The other may be C4 (order) as a gate:
  first show graded order, then test bending.
- **Design size.** The smallest Q × L × r with oracle power ≥ .8 at θ = 60, in q mode, at the measured persona SNR on the `syc`/`opt`
  background. If none is reachable within n ≤ 1000 per level, the persona curvature study is declared infeasible at that layer.

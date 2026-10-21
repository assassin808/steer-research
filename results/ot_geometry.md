# Is steering space flat? Trait flows (OT as a baseline) on IDEEA's head activations — Gemma-2-2B, TruthfulQA

Status: **v1 draft, before adversarial review.**
Prereg: `results/ot_geometry_prereg.md`, with these amendments and deviations:
- A1: power-driven switch to a smoothed OT statistic;
- A2: trait-flow reframe;
- D1: OT spaces cut;
- D2: post hoc long horizon and α-grid extension.

Code: `ot_geometry/`. Summaries:
- `results/trait_flow_summary.json`
- `results/trait_flow_long_summary.json`
- `results/ot_geometry_summary.json`

Raw outputs are in `~/scratch/ot_geom/`. CPU only, no GPU job.

## Setup
| | |
|---|---|
| Data | IDEEA cache `head.pkl` (o_proj input, last token), development split only: 395 questions, 3049 rows (1415 POS, 1634 NEG) |
| Fit set | IDEEA `train` questions, 2427 rows |
| Held-out set | IDEEA `val` questions, 79 questions, 622 rows, of which 335 NEG start points |
| Test questions | never loaded |
| Heads | IDEEA top-24 (ITI val-accuracy ranking, paper K = 24) + 8 random controls |
| Space | z (per-coordinate standardised, 256-d) is primary; pw32 / pw8 (PCA-whitened) are sensitivity |

**Objects.**
- Trait score s = log-ratio classifier fitted on unpaired rows: L2 logistic regression (LR) and a 64-unit tanh MLP, with grouped CV over questions.
- Trait flow V = P_T∇s / |P_T∇s|, where P_T is the local PCA of the 64 nearest fit points at ≥ 90% variance (median rank 13–16 of 256).
- The unprojected flow is V₀ = ∇s/|∇s|.
- Paths are integrated by Euler from held-out NEG points, with step 0.02·|Δμ|.

**Limits checked in code.**
- Linear s gives a constant V₀: the LR turning is exactly 0.0°, which is the ITI/probe-like flat field.
- IDEEA `nearest_pos_neg` is a hard-assignment, piecewise-constant field.
- The OT barycentric displacement is a different object.

## 1. Is s nonlinear? (gate)
| run | space | held-out AUC LR → MLP (median) | MLP log-loss better (CI > 0) | MLP worse (CI < 0) |
|---|---|---|---|---|
| **prereg** (α grid ≤ 10; CV picked α = 10 = grid edge on every head) | z | .844 → .859 | 0/24 | 17/24 |
| prereg | pw32 | .801 → .844 | 4/24 | 0/24 |
| **post hoc D2** (α grid to 300; CV picks interior α = 100 on 22/24, α = 30 on 2) | z | .844 → .854 | **15/24** | 0/24 |
| post hoc D2, random heads | z | .714 → .720 | 3/8 | 1/8 |

The preregistered MLP was under-regularised: it was overconfident, with worse log-loss even though its AUC was higher. With adequate weight decay, a nonlinear log-ratio is supported on held-out questions on 15/24 top heads. The effect is small: median +0.013 nats per row and +0.01 AUC.

## 2. Flatness: do flow lines turn?
Statistic: turning angle ∠(V(x₀), V(x_ℓ)) after arc length ℓ = |Δμ|, as the median over held-out starts. Null bands come from refitting the same pipeline on equal-covariance Gaussians (G-shared, where the true field is constant) and on label permutations.

| run | flow | data | G-shared null | perm null | heads "curved" (> G-shared 97.5%) |
|---|---|---|---|---|---|
| prereg α = 10, z | unprojected | 28° | 19° | 31° | 24/24 (random 8/8) |
| prereg α = 10, pw32 | unprojected | 17° | 19° | 27° | 0/24 |
| prereg, z / pw32 | projected | 39° / 35° | 42° / 36° | – | 0/24 / 3/24 |
| **post hoc α = 100, z** | unprojected | **10.1°** (17.0° at 3|Δμ|) | **0.03°** (0.27°) | – | **24/24** (random 6/8) |
| post hoc α = 100, z | projected | 35° | 41° | – | 0/24 |

**Verdict against the prereg rule: "flat by Occam".** In z the nonlinearity gate failed, and in pw32 0/24 heads are curved. The α = 10 estimator is dominated by its own noise: the null alone turns 19°, and in z the data turning (28°) sits near the permutation null (31°).

**Post hoc, the conclusion reverses.** With the CV-optimal regularisation, the Gaussian-null field is linear to within 0.03°, while the data field turns about 10° per |Δμ| (17° by 3|Δμ|) on every top head and on 6/8 random heads. The nonlinearity behind it generalises to held-out questions on 15/24 heads. So the field is mildly curved: ≈10° of turning over the distance between class means.

The projected flow cannot resolve this. Tangent-estimation noise alone turns 34° (linear s, projected), which exceeds the signal.

## 3. Estimator agreement with the local field
Mean cosine with V₀(x) at held-out NEG points, prereg run, top-24 median.

| | z | pw32 |
|---|---|---|
| linear-probe weight w | **.78** (null .91) | .72 |
| mean difference Δμ | .43 (null .42) | .72 |
| IDEEA nearest_pos_neg nc = 2 | .34 | .54 |
| OT displacement | .10 | .23 |
| self-consistency cos(V, mean V) | .92 (null .96; perm .75) | .76 |

Findings:
- The prereg prediction holds: w > Δμ ≥ IDEEA > OT.
- In z, w and Δμ differ (cos .63) because of the within-class covariance. In pw32 they coincide (.99) and both match V equally.
- IDEEA's direction is mostly the mean difference (cos .68 with Δμ). It does not track the local field better than Δμ does: agreement .34 vs .43.
- The OT displacement is nearly unrelated to the trait gradient field, which is expected: it is a different flow.

## 4. Boundary
Post hoc horizon to 30·|Δμ|, z, top-24 medians, arc length in units of |Δμ|. The band is the 10-NN distance to fit POS ≤ the 95% quantile for held-out POS. About 96% of NEG starts are already inside the POS band: in 256-d the class supports overlap.

| path | leaves POS band | s saturates (ds/dℓ < 10%) |
|---|---|---|
| straight along Δμ | **6.5** | 6.5 |
| straight along probe w | 5.6 | 4.6 |
| trait flow, unprojected | 6.4 | 3.6 |
| trait flow, projected | **10.8** | 11.6 |
| linear-s flow, projected | 8.75 | – |

What this shows:
- **Prediction "the straight line leaves the support sooner than the projected flow": supported** on 24/24 heads, and point-wise for 98% of starts.
- Most of the gain is the projection itself. The linear-s projected flow exits at 8.75; the learned curvature adds 8.75 → 10.8.
- The unprojected flow is no better than a straight line (6.4 vs 6.5). It turns, but not toward the support.
- On every head, s saturates before the support exit along the unprojected flow (3.6 vs 6.4). Caveat: a tanh-MLP logit is bounded, so "saturation" is partly a model property.

**Link to IDEEA's behaviour (no GPU).** IDEEA adds α·naive_std·v̂ per head, which equals arc length t = α·naive_std/|Δμ_raw|:
- α = 4 (paper config, official TI .572) ≈ 5.0|Δμ|, median over top heads. It is inside the straight-line support exit on 20/24 heads.
- α = 15 (CLI default, TI .099 < base) ≈ 18.7|Δμ|. It is outside on 24/24 heads.

So the paper's working strength sits just inside the support boundary and the failing default sits about 3× past it. Caveats: IDEEA steers 24 heads at once along its own v̂ (not Δμ), and the effects compound across layers. This is a consistency check, not a causal test.

pw32 agrees on the straight vs projected ordering (5.75 vs 9.6). There, s does not saturate before the exit.

## 5. OT baseline column (A1 statistic R²_s50, primary ε = 0.01·median cost)
| | z top-24 | pw8 top-24 |
|---|---|---|
| R²_s50 data | .50 | .69 |
| null: R-shift (data-shape translation) | .81 | .93 |
| null: G-shared | .64 | .92 |
| null: G-sep (affine) | .53 | .79 |
| null: perm | .21 | .15 |
| heads curved vs both translation nulls | 22/24 (12 explainable as affine) | 24/24 (all beyond affine) |

Also:
- Robust across ε: 22/24 at c ≤ 0.03, 12/24 at c = 0.1 (near-uniform plan).
- Raw R² cannot tell anything apart (0.06 vs 0.07), as the power check predicted.
- Held-out OT-displacement predictors (z): kNN-local .39, mean diff .22, IDEEA npn .19, min_perp .19.
  - Under the R-shift null (truth constant) the same predictors score .42 / .34. The kNN advantage is therefore partly a smoothing artefact. The real signal is that mean diff falls far below its own ceiling (.22 vs .34).
  - IDEEA beats mean diff on 0/24 heads.
- Geodesic/Euclid ratio is uninformative in 256-d (1.00 everywhere; the partner is a graph neighbour). In pw8 it is 1.24 vs 1.00 under the null, curved on 16/22 heads.

Power (synthetic, real n, 6 null reps):
- In 256-d, R²_s50 detects a mode-dependent field once |Δμ| = 3 and the field's true R² ≤ 0.85.
- At |Δμ| = 1 it reports 1/4 flat configurations falsely as curved (marginal).

## Post-hoc: de-facto question pairing (not preregistered)
- Per-question δ_q = mean POS_q − mean NEG_q, with a split-half noise correction.
- The constant share is .13 on top heads (range .05–.18) and .01–.10 on random heads.
- δ_q mixes truth with answer-content differences, so this is a lower bound on the constant share, not a curvature measurement.

## Bottom line (v1)
1. **Prereg verdict: flat by Occam.** The preregistered estimator could not show a held-out-supported nonlinear trait score in z, and its turning did not exceed the null in pw32.
2. **Post hoc, with properly regularised s: weakly curved.**
   - The nonlinear log-ratio generalises (15/24 heads, +0.013 nats).
   - The field turns about 10° per |Δμ| against a null of about 0°.
   - The linear probe direction stays the best global summary (cos .78 with the local field); IDEEA's clusters add nothing beyond Δμ.
   - "Flat to first order, curved at second order" is the defensible framing, pending replication on fresh questions.
3. **Boundary (paper C).**
   - Straight extrapolation leaves the POS support at about 6|Δμ|; tangent projection extends this to about 11|Δμ|.
   - IDEEA's good α sits inside this boundary and its failing α far outside.
4. **OT** sees curvature as well (22/24 heads curved), but that is a statement about the transport map. It barely aligns with the trait gradient field (cos .10).

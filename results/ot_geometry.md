# Is steering space flat? Trait flows (OT as a baseline) on IDEEA's head activations — Gemma-2-2B, TruthfulQA

Status: **v2, revised after the adversarial Opus review.** v1 made a post-hoc "weakly curved" claim; it is withdrawn here.

Prereg: `results/ot_geometry_prereg.md`, with these amendments and deviations:
- A1: power-driven switch of the OT statistic;
- A2: trait-flow reframe;
- D1: OT spaces cut;
- D2: long horizon and α-grid extension;
- D3: shape-matched nulls;
- D4: boundary controls.

Everything after A2 that used results is marked **POST HOC**.

Code: `ot_geometry/`. Summaries:
- `results/trait_flow_summary.json`
- `results/trait_flow_long_summary.json`
- `results/trait_flow_tilt_summary.json`
- `results/trait_flow_bdctl_summary.json`
- `results/ot_geometry_summary.json`

Raw outputs are in `~/scratch/ot_geom/`. CPU only, no GPU job.

## Setup
| | |
|---|---|
| Data | IDEEA cache `head.pkl` (o_proj input, last token), development split only: 395 questions, 3049 rows (1415 POS = correct reference answers, 1634 NEG) |
| Fit set | IDEEA `train` questions, 2427 rows |
| Held-out set | IDEEA `val` questions, 79 q, 622 rows, 335 NEG starts |
| Test questions | never loaded |
| Heads | IDEEA top-24 (ITI val-accuracy ranking, K = 24) + 8 random |
| Caveat on heads | the ranking used the same val questions, which favours LR and the agreement numbers; random controls are weak (4/8 come from layers 0–1) |
| Space | z (per-coordinate standardised, 256-d) is primary; pw32 / pw8 (PCA-whitened) are sensitivity |

**Objects.**
- Trait score s = log-ratio classifier from unpaired rows: L2 logistic regression (LR) and a 64-unit tanh MLP, with grouped CV by question.
- Trait flow V = P_T∇s/|P_T∇s|, with P_T the local PCA of the 64 nearest fit points at ≥ 90% variance (median rank 13–16).
- Unprojected flow V₀ = ∇s/|∇s|.
- Euler steps of 0.02|Δμ| from held-out NEG.

**Limits checked in code.**
- Linear s gives a constant V₀: the LR turning is exactly 0.0° (ITI/probe-like).
- IDEEA nearest_pos_neg is a piecewise-constant field.
- OT displacement is a different flow.
- Important (stated in A2): a translation of a non-Gaussian shape does **not** give a linear log-ratio. So "trait-flow flat" must be tested against a shape-matched null, not only against Gaussians.

## Headline
1. **Verdict (prereg rule): FLAT — no curvature detected.** This is "flat by Occam" in z (the nonlinearity gate failed) and flat on 24/24 heads in pw32.
2. **The post-hoc curvature in v1 was wrong.** With better-regularised s (α = 100), the flow turns about 10° per |Δμ|. That is far above a Gaussian null (0.03°) but **below** the null where POS is a pure translation of the real data shape (R-shift, 14°). The data exceeds that null on only **2–3/24 heads**. The non-Gaussian shape alone explains the turning.
3. **Estimators.** Local field vs global directions: probe weight w .78 > Δμ .43 > IDEEA .34 > OT .10. IDEEA's clusters do not track the local trait field better than Δμ does.
4. **Boundary (paper C): largely not trait-specific.** Along +Δμ the path leaves the data support at 6.5|Δμ|, only about 1.25|Δμ| later than random directions (5.0; +Δμ beats all of them on 24/24 heads, but random directions inside the top-10 PCs on only 18/24). Measured by Euclidean displacement, the projected trait flow ties with the straight line (paired median +0.07|Δμ|, later on 12/24 heads; within grid resolution). The prereg prediction "the straight line leaves the support sooner" therefore fails. The IDEEA α↔t link is only an order-of-magnitude consistency check.

## 1. Is s nonlinear? (gate)
| run | space | held-out AUC LR → MLP (median) | MLP log-loss better (CI > 0) | worse (CI < 0) |
|---|---|---|---|---|
| prereg (α grid ≤ 10; CV picked the grid edge α = 10 on every head) | z | .844 → .859 | 0/24 | 17/24 |
| prereg | pw32 | .801 → .844 | 4/24 | 0/24 |
| POST HOC D2 (α to 300; CV picks interior α = 100, or 30 on 2 heads) | z | .844 → .854 | 15/24 | 0/24 |

How far the D2 result goes:
- It shows only that the log-ratio is not exactly linear, i.e. the class-conditionals are not equal-covariance Gaussians.
- The effect is small: +0.013 nats per row, +0.01 AUC.
- The 24 heads are not independent: they share 79 held-out questions, and many sit in the same layers.
- The reviewer measured the share of the MLP−LR residual explained by question (η²) at 0.25–0.39 vs ≈0.13 at chance. The residual correlates with each question's POS rate (r ≈ 0.2–0.34), so part of the gain is question-level base rate. Within-question AUC still improves slightly.
- This does **not** imply a curved trait field.

## 2. Flatness: turning of flow lines (unprojected; angle after arc length |Δμ|, median over held-out starts)
| run | data | G-shared (Gaussian, field constant) | R-shift (real shape, pure translation) | Perm | heads above null |
|---|---|---|---|---|---|
| prereg α = 10, z | 28° | 19° | 23° | 31° | 24/24 vs G-shared (verdict gated off: s not nonlinear) |
| prereg α = 10, pw32 | 17° | 19° | 16° | 27° | **0/24** vs G-shared |
| POST HOC α = 100, z | 10.1° | 0.03° | **14.0°** | – | 24/24 vs G-shared, but **2/24 above the R-shift max** (reviewer, independent reps: 3/24) |
| POST HOC α = 100, z, tilted-empirical null (linear log-ratio on resampled real rows) | 10.1° | jitter 0.1: 22°; jitter 0.5: 2.9° | – | – | 2/24 / 24/24 — uninformative, depends on the smoothing bandwidth |

Notes:
- **Projected flows** turn about the same as their null (39° vs 42° in z). Tangent-estimation noise dominates: linear s alone turns 34° after projection.
- **Metric caveat.** Turning angles depend on the metric: z vs pw32 disagree at α = 10, and pw32 also drops 224 dimensions. The principled choice is a full-dimensional whitened (Fisher-under-Gaussian) metric. It was not run.
- **Null reps.** G-shared uses 5–10 reps, so the 97.5% quantile is effectively the maximum, and lbfgs occasionally lands on a 2–5° solution. A replication needs ≥ 20 reps.

**Interpretation.** Within what 2.4k unpaired rows in 256-d can resolve, the truth flow is indistinguishable from what a constant shift of the data's own shape would produce. In the transport sense that means flat; the non-flatness of ∇ log p+/p− is inherited from data shape, not from a trait-specific bend. Paper W should state the result this way. A preregistered replication on fresh questions (α = 100, R-shift primary, 20 reps, full whitened metric) is needed before claiming "flat" positively.

## 3. Estimator agreement (mean cos with V₀(x) at held-out NEG; prereg run, top-24 median)
| | z | pw32 |
|---|---|---|
| linear-probe weight w | **.78** (same pipeline under G-shared: .91) | .72 |
| mean difference Δμ | .43 (G-shared .42) | .72 |
| IDEEA nearest_pos_neg nc = 2 | .34 | .54 |
| OT displacement | .10 | .23 |
| self-consistency cos(V₀, mean V₀) | .92 (G-shared .96, Perm .75) | .76 |

Notes:
- The prediction w > Δμ ≥ IDEEA > OT holds.
- In z, Δμ differs from w (cos .63) because of the within-class covariance. Strictly, ∇s and w are covectors and Δμ is a vector; their cosine is metric-dependent. In pw32 they coincide (.99).
- IDEEA's direction is mostly Δμ (cos .68).
- The OT displacement is a different object.
- Data values are about as far below the Gaussian-null values as the null spread allows. This is not evidence of curvature.

## 4. Support boundary (POST HOC horizon to 30|Δμ|; z; top-24 medians; units of |Δμ|)
About 96% of NEG starts already lie inside the "POS band" (10-NN distance to fit POS ≤ the q95 of held-out POS). So the band measures **data support**, not POS support. The band radius is about 21–25 z-units, against |Δμ| ≈ 2–4.4.

| path | exit (arc length) | exit (Euclidean displacement) |
|---|---|---|
| straight +Δμ | 6.5 | 6.5 |
| straight probe w (old run, `trait_flow_long_summary.json`) | 5.6 | 5.6 |
| straight −Δμ | 4.9 | 4.9 |
| straight, 5 random directions | 5.0 | 5.0 |
| straight, 5 random directions in the top-10 PCs | 5.75 | 5.75 |
| trait flow, unprojected | 6.4 | 5.9 |
| trait flow, projected | 10.7 | 7.0 |
| linear-s flow, projected | 8.75 | 6.9 |

**Reading (POST HOC D4 controls, all 24 top heads; `results/trait_flow_bdctl_summary.json`).**
- **The boundary is mostly not trait-specific.** Any straight direction leaves the data support after about 5|Δμ|:
  random directions 5.0, −Δμ 4.9, random directions inside the top-10 PCs 5.75.
  +Δμ goes only about 1.25|Δμ| further (6.5). It beats all 5 random directions on 24/24 heads but all 5 top-PC
  random directions on only 18/24. So the "boundary" is essentially the kNN radius of the data cloud
  (band ≈ 21–25 z-units vs |Δμ| ≈ 2–4.4). The modest +Δμ advantage is consistent with stepping toward the denser
  POS side of an overlapping cloud.
- **Prereg prediction "the straight line leaves the support sooner than the projected flow" FAILS under a fair measure.**
  By arc length it holds on 24/24 heads (10.7 vs 6.5), but that is a turning artefact: a path that turns covers less
  ground per unit arc. By Euclidean displacement the projected trait flow and straight +Δμ are tied: medians 7.0 vs 6.5
  (a difference of medians), paired per-head median difference +0.07|Δμ|, flow later on only 12/24 heads.
  That gap is below the resolution difference (straight lines on a 0.25|Δμ| grid, flows on 0.1|Δμ| steps; the bias
  favours the straight line). The linear-s projected flow gives the same (6.9).
  The unprojected flow exits earlier than +Δμ (5.9; 24/24 heads), close to the straight probe-w line (5.6).
  The gap reflects where it points, not anything gained by following a flow.
- A band built from all data (not POS only) gives the same conclusion, more strongly: +Δμ 6.25, random 5.25,
  top-10-PC random 5.9; +Δμ beats all top-10-PC directions on only 14/24 heads.
- Random heads: straight lines exit about 3× later (flow medians there are biased low: 46% of projected paths never exit and are dropped) (+Δμ 18, random 15.6) because Δμ is small relative to the cloud there.
- For paper C: in this activation geometry the support boundary is set by the overall data scale, not by the trait.
  Neither following the trait flow nor tangent projection buys support-preserving distance over a straight line.
  A trait-specific "reach" must be defined behaviourally (KL / judged answers along t), not by kNN support.

**IDEEA α ↔ t.** IDEEA adds α·naive_std·v̂ per head. Along Δμ that corresponds to t = α·naive_std/|Δμ_raw|:
- α = 4 ≈ 5.0|Δμ| (median; range 2.6–7.6 across heads);
- α = 15 ≈ 18.7|Δμ|.

This is only order-of-magnitude consistent with α = 4 working and α = 15 failing:
- α = 4 is the paper config: strict TI .300 vs base .193.
- α = 15 is the CLI default: official TI .099, below base.

Reasons not to read more into it:
- IDEEA's v̂ is not Δμ (cos .68).
- IDEEA steers 24 heads at once, last token only, and effects compound across layers.
- ITI, which steers along Δμ, does not beat base under strict labels (.170).
- Behaviour was not measured along t.

The "s saturates before exit" statistic from v1 is dropped: a tanh-MLP logit is bounded.

## 5. OT baseline column (A1 statistic R²_s50, ε = 0.01·median cost)
| | z top-24 | pw8 top-24 |
|---|---|---|
| R²_s50 data | .50 | .69 |
| R-shift null (data-shape translation) | .81 | .93 |
| G-shared | .64 | .92 |
| G-sep (affine) | .53 | .79 |
| Perm | .21 | .15 |
| curved vs both translation nulls | 22/24 (12 affine-explainable) | 24/24 |

What this column shows:
- OT does see a non-constant transport map, even against the shape-matched null. This is a statement about the minimum-kinetic-energy coupling of NEG to POS, which differs from the trait gradient field (cos .10).
- One possible reading is that the "curvature" seen by OT reflects how POS and NEG are mixed across questions, not a trait-specific bend. Untested.
- Robust over ε ≤ 0.03·median cost; it weakens at 0.1 (near-uniform plan).
- Raw R² has no power, as predicted by the synthetic check.
- Held-out predictors of the OT displacement: kNN-local .39, Δμ .22, IDEEA npn/min_perp .19.
  - Under the R-shift null the kNN-local and Δμ predictors score .42 / .34, so the kNN advantage is partly smoothing.
  - IDEEA beats Δμ on 0/24 heads.
- The geodesic/Euclid ratio is degenerate in 256-d (1.00); in pw8 it is 1.24 vs 1.00 (curved on 17/24 heads).
- Power (synthetic, real n): R²_s50 detects a mode-dependent field at |Δμ| = 3 when the field's true R² ≤ 0.85. At |Δμ| = 1 it gave 1/4 false positives.

## Post-hoc: question pairing (not preregistered)
- Per-question δ_q = mean POS_q − mean NEG_q, with a split-half noise correction.
- Constant share .13 on top heads (range .05–.18) and .01–.10 on random heads.
- δ_q mixes truth with answer-content differences, so this is a lower bound on the constant share, not a curvature measurement.

## Adversarial review (Opus) — objections and handling
| # | objection | severity | handling |
|---|---|---|---|
| 1 | α = 100 "curvature" is fully explained by the R-shift null (reviewer: 3/24 above; multivariate-t null 8.6°) | fatal for the v1 claim | Claim withdrawn. Our own D3 run gives 2/24. Headline is now flat / not detected. |
| 2 | The gate only shows non-Gaussian class-conditionals; heads are not independent; small effect | major | Reworded (§1). |
| 3 | G-shared null too few reps and unstable | major | Stated; the replication spec requires ≥ 20 reps. |
| 4 | Metric dependence of the turning angle; pw32 also truncates dimensions | major | Stated; full-dim whitened metric recommended, not run. |
| 5 | Question clustering / base rates in the MLP gain | moderate | Reviewer's η² and within-question AUC reported (§1). |
| 6 | Head-selection bias (val questions); weak random controls | minor | Stated in Setup. |
| 7 | Forking paths (α grid extended after the gate failed) | major | All post-hoc steps labelled; nothing post hoc is a headline claim. |
| 8 | The "support boundary" is mostly a kNN-radius effect; random directions exit at ≈4.5 vs ≈6 for Δμ | fatal for the v1 framing | Renamed "data support"; random/−Δμ controls on all heads (D4) in §4. |
| 9 | Arc-length confound for the projected flow | major | Euclidean-displacement exits added (D4). |
| 10 | α ↔ t over-read; official vs strict TI | major | Downgraded to order-of-magnitude; strict numbers used. |
| 11 | Saturation meaningless (tanh bound) | moderate | Dropped. |
| 12 | Code | – | No invalidating bug found. G-shared/R-shift starts are drawn from the null (Perm uses real starts); cos(covector, vector) noted. |

## Not done
- Composition: no second label set is cached; it would need a GPU extraction.
- Behaviour along t (realised KL / answers vs α).
- Full-dim whitened metric.
- Replication on fresh questions.

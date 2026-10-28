# Weekday/month calibration of the geometry toolkit — Gemma-2-2B-it

Status: **v1, final pre-review** (review section below). Prereg: `results/weekday_calib_prereg.md` (commit 910da50).
Code: `weekday/` (`extract.py`, `geom.py`, `run_geom.py`, `old_stats.py`, `fourier.py`, `aggregate.py`, `table.py`, `old_table.py`).
Raw outputs: `~/scratch/weekday/`. Summary: `results/weekday_calib_summary.json`. Figures: `results/weekday_calib_*.png`.
Compute: CPU only. Narval GPUs were in the POWEROUT20261027 reservation until 2026-11-03, so GPU job 7499750 was cancelled.
The run used about 2 h of login-node CPU for extraction and steering.

## Headline
1. **The preregistered toolkit FAILS the positive control.** It fails on months (the prereg gate) and on weekdays, and the 3×3 tuning grid
   (k ∈ {5, 10, 20}, PCA ∈ {16, 64, 256}) does not rescue it. In detail:
   - **TwoNN ID** is 3.6–8, never 1–2.
   - **Geodesic distortion** never exceeds the SNR-matched straight-line null at the last token. At the z token it does on only 9/13 layers.
   - **Distance cyclicity** of class d_G beats shuffle on ≤ 8/13 layers.
   - **Centroid turning** sits at 100–120° at every position. That is the value for a regular simplex (pure identity, 120°), not for a smooth 7- or 12-gon (51°/30°).
2. **Known cyclic structure is present and measurable**, but only with order-aware class-level statistics.
   **These statistics are post hoc** (A4, S7 adjacency; S8 first circular harmonic of the split-half centroid Gram):
   - With z (start-day/month) labels the f1 power is 8–200× the label-shuffle maximum at every layer.
   - The f1 share is 0.4–0.7, against a pure-simplex level of 0.33 (weekdays) and 0.18 (months).
   - In the top-2 reliable subspace the cyclic order is perfect (adj = 1.0) at most z-token layers.
   - The representation is therefore **mostly categorical (simplex) plus a cyclic component**, not a 1-D curved manifold.
3. **Several readouts never had power** here.
   - The ot_geometry statistics O3 (geo_ratio, 1.000 on the two-class graph) and the OT R²_s50 do not separate data from the R-shift null in an interpretable way.
   - O1 flow-line turning at the TruthfulQA setting (α = 100) beats the G-shared and R-shift nulls on 0–4/24 pairs, against about 1–2 expected by chance; one exception is 8/24 (month z, L8).
   - With CV-selected α, O1 beats both nulls on 4–12/24 pairs for z labels, i.e. partial power.
   - O2's target-only MMD profile (d* ≈ 0.5–1.0|Δμ|) looks the same as on TruthfulQA.
   - The intermediate-class MMD (straight chord vs centroid polyline, held-out intermediates) is lower along the polyline on 14/14 and 24/24 pairs at the last token. This is the only old-pipeline readout that sees the bend, and it needs the intermediate classes.
4. **The TruthfulQA "flat" verdict carries little evidential weight.**
   - TruthfulQA top-24 heads have class SNR 0.004–0.018 (centroid power / C / within-class variance, same definition).
   - The weekday/month last-token z labels are at 0.005–0.05 in mid layers. At that SNR the preregistered statistics mostly do not detect a cycle we know is there.
   - The old binary statistics also cannot represent "curvature through intermediate states": with two classes there is nothing in between.
   - "Flat" should be restated as **"no curvature detectable with tools shown to lack power at this SNR"**, not as evidence of flatness.
5. **Steering (optional) is UNINFORMATIVE.**
   - Answer-label centroid differences at L8–20 are tiny (class SNR ≈ 1e-4 before L20).
   - Adding the full Mon→Wed difference leaves the argmax at Monday for all t, with P(target) ≤ 0.03 at t = 1, on both straight and spline paths.
   - There is no Tuesday bump to compare, and off-class mass stays < 1%.
   - Spline ≥ straight on the intermediate probability at t = 0.5 holds only at L20, and only by 0.003–0.03, so the prereg pass rule is not met in any meaningful sense.
   - Reproducing Wurgaft's figure would need late layers (≥ 21, where the answer circle appears) or larger scale. Not run.

## Data
| | weekday | month |
|---|---|---|
| prompts | 7 z × 13 k × 6 templates = 546 | 12 × 11 × 6 = 792 |
| accuracy (top-1 among class tokens) | .28 (k=1,2,3: 1.00/1.00/.79; k≥4 ≤ .21) | .53 (k=1: 1.00, k=2: .92, then .18–.68) |

Positions:
- last prompt token ("The answer is **", amendment A1);
- the z token (A6, post hoc). The z-token prefixes are identical across k for 2 templates, so the data is deduplicated to 378 / 552 points.

Labels:
- **y** = correct answer (prereg primary);
- **z** = start day/month;
- pred = the model's answer;
- correct-only subset.

**Confound for y (found after the data was in).** Class y's (z, k) multiset is a cyclic shift of class 0's. Any z-encoding therefore makes
y-centroids circulant, i.e. cyclic, even if the model never computes the answer. At mid layers, y-centroid power is 1e-4 of the within-class
variance and is smaller than shuffled-label power: the label is not computed there. y is interpretable only in late layers (≥ 21), where it
becomes strong and nearly circular (weekday f1 .84). The cleaner test is **z labels**.

## Amendments (all after the prereg commit)
- **A1** (technical, before data): forced prefix "The answer is **"; no-space class tokens. Qwen2.5-1.5B weights incomplete → second model not run.
- **A2** (synthetic, before data): the mean turning of any closed polygon is ≥ 360/C (a back-and-forth line also gives 51°) → read the **median**.
- **A3** (estimator bug, after data): random within-class split halves have complementary (z, k) composition, which gives negative cross-Gram eigenvalues → split by template (3 of 6).
- **A4** (POST HOC): S7, cyclic order in the split-half top-2 subspace (adj).
- **A5** (POST HOC): line2 null, an SNR-matched unrolled line: equally spaced centroids on one direction, the data's centroid spread, permuted real residuals. The prereg "line" null (centroids projected on their top PC) folds the cycle and has lower SNR than the data.
- **A6** (POST HOC): z-token position as a high-SNR control.
- **A7** (CPU budget): steering on layers 8/12/16/20 with straight and spline paths only (the polyline was dropped).
- **S8 Fourier** (POST HOC): first circular harmonic share of the template-split centroid Gram, with a within-template label-shuffle null.

## Results, preregistered statistics
"x/n" = layers where the data beat every null rep. The prereg middle window is 6–18. The late window, 19–25, is POST HOC: structure is strongest there.

| run (mid layers) | ID | distortion vs line2 | cyc_G vs shuffle | turning (median) vs line2 | top-1 share vs line2 | adj vs shuffle |
|---|---|---|---|---|---|---|
| month y (prereg gate) | 6.3 | 1/13 | 7/13 | 120° 0/13 | .41 7/13 | .42 6/13 |
| month z | 6.3 | 2/13 | 1/13 | 107° 0/13 | .36 12/13 | .58 10/13 |
| weekday y | 6.9 | 2/13 | 4/13 | 99° 0/13 | .49 13/13 | 1.0 11/13 |
| weekday z | 6.9 | 4/13 | 0/13 | 103° 2/13 | .45 11/13 | .71 7/13 |
| z token, weekday z | 4.5 | 9/13 | 1/13 | 108° 13/13 | .34 13/13 | 1.0 12/13 |
| z token, month z | 3.7 | 9/13 | 2/12 | 112° 13/13 | .22 13/13 | 1.0 13/13 |

Late layers (19–25), month: ID drops to **2.5**; turning beats line2 6–7/7; adj 1.0.
Late ID ≈ 2.5 is the only case near the predicted 1–2. It coincides with the output layers, where activations collapse toward the answer
logits, so it is not obviously the manifold.

Other statistics:
- **Distortion vs the flat Gaussian**: always above (13/13), but **equal to the line null** at the last token. The kNN-graph distortion comes from the label-free nuisance shape (prompt content), not from the class arrangement.
- **Synthetic check**: on a noisy synthetic circle, distortion is 0.03 and TwoNN is 26. A circle's chord is monotone in arc length, so 1 − ρ(d_G, d_E) has no power for circles by construction.
- **Local-field rotation** is always more coherent than shuffle (13/13) but never beats line2.
- **Tuning (months, 9 settings, y and z)**: no setting gives ID ≤ 3 in mid layers. PCA-16 lowers ID to 4.6. Distortion vs line2 stays ≤ 6/13, turning vs line2 ≤ 4/13, and cyc_G vs shuffle ≤ 8/13. **Verdict FAIL is robust to tuning**, so no setting is available to "confirm on weekdays".
- **Correct-only subsets** (n = 154 / ~420): nothing beyond the full-data pattern; weekday correct detects nothing.

**Prereg verdicts.**

| prediction | result |
|---|---|
| P1 (ID ≤ 3, mid) | FAIL |
| P2 (cyclicity d_G > shuffle; d_G ≥ d_E) | FAIL (mid ≤ 8/13); d_G ≥ d_E mostly holds for y only |
| P3 (distortion > flat-G and > line) | FAIL (line) |
| P4 (turning ≥ 25° and above every line rep) | FAIL vs line; the top-1 share part passes |
| Toolkit gate | **FAIL** |

## Results, Fourier / order statistics (POST HOC, `weekday_calib_fourier.png`)
| run | mid f1 share | lin share | f1 power > shuffle max | class SNR (mid) |
|---|---|---|---|---|
| weekday z, last token | .62 (simplex .33) | .37 | 13/13 | .011 |
| month z, last token | .42 (simplex .18) | .23 | 13/13 | .010 |
| weekday z, z token | .53 | .28 | 13/13 | 1.8 |
| month z, z token | .39 | .19 | 13/13 | 1.8 |
| weekday y, late | .84 | .26 | 5/7 | .054 |
| month y, late | .71 | .23 | 7/7 | .30 |

Reference: TruthfulQA top-24 heads, class SNR .004–.018 (median .009).

## Old statistics (ot_geometry.md / klflow.md) on class pairs i→i+2, i→i+3
Layers 4/8/12/16/20/24, 10 null reps (so "beat the max" has a chance rate of ≈ 1/11 per pair).
- **O1, α = 100** (the TruthfulQA setting): 0–8/24 pairs above both the G-shared and R-shift nulls, most often near chance.
  The MLP is essentially linear (turning ≈ 0).
- **O1, CV α**:
  - z labels: 4–12/24 above both nulls → partial power.
  - y labels: 14/14 and 23/24 at L4–8, where y carries no answer signal → this reflects the composition confound, not curvature.
- **O2**:
  - d* (target MMD minimum) is 0.5–1.0|Δμ| everywhere, the same form as on TruthfulQA. The target-only diagnostic is uninformative about bends.
  - Intermediate-class MMD (held out), z labels: polyline < straight on 14/14 pairs (weekday) and 24/24 pairs (month) at the last token, relative gap 8–120%.
  - At the z token the gap is small (0–10%; weekday 5–12/14), because the simplex part dominates: intermediates are about as far from the chord as from anything else.
- **O3 geo_ratio**:
  - two-class graph: 1.000 at the last token (degenerate, as in TruthfulQA);
  - all-class graph: 1.0–2.3;
  - z token: 1.1–5.7, but with no null this is a scale effect, not evidence.
  - **OT R²_s50** < every R-shift rep on 0–24/24 pairs, with no consistent pattern (y: 24/24 driven by composition) → uninterpretable.

## Steering (A7; 13 weekday / 11 month prompts; sanity check: t = 0 reproduces unsteered probabilities to < 1.3e-6)
| layer | Mon→Wed P(Tue) at t = .5, straight / spline | P(Wed) at t = 1 | Jan→Mar P(Feb) at .5 | P(Mar) at t = 1 |
|---|---|---|---|---|
| L8–16 | .028 / .028–.029 | .01 | .092–.098 / .092–.102 | .01–.02 |
| L20 | .031 / .034 | .02 | .137 / .166 | .03 |

- Off-class mass ≤ .009 everywhere; argmax = source on every t.
- Both paths are too small to move the answer: the class means of the answer label are nearly identical before L21.
- **UNINFORMATIVE**, not a failure of Wurgaft's claim.

## What this means for paper W
- A "flat" verdict from ID, distortion, centroid turning or the old flow/OT statistics has **no demonstrated power** at TruthfulQA-like SNR.
  At that SNR these tools do not see a cycle we know exists.
- Order-aware class statistics (adjacency, circular Fourier share with label-shuffle nulls) do see it at SNR ≈ 0.01. Those tools need ≥ 3 ordered classes.
- For a binary truth label, "curved vs flat" is not well posed without intermediate states. The TruthfulQA test should be re-posed with graded labels
  (e.g. intensity levels) before it can say anything.
- The weekday/month representation itself is mostly categorical: turning ≈ 108–112° at the z token is closer to a simplex's 120° than to a 7/12-gon's 51/30°.
  Any "manifold steering" claim needs to say which subspace it means.

## Not done
- Qwen2.5-1.5B (weights incomplete offline).
- Steering at late layers (≥ 21) or at larger scale.
- Old O1–O3 nulls with ≥ 20 reps.
- The old weekday-y run stopped after L12 on a Lustre I/O error (y is confounded anyway).

## Adversarial review
(pending)

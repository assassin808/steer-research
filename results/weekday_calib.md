# Weekday/month calibration of the geometry toolkit — Gemma-2-2B-it

Status: **v2, revised after the adversarial Opus review** (review table at the end). Prereg: `results/weekday_calib_prereg.md` (commit 910da50).
Code: `weekday/` (`extract.py`, `geom.py`, `run_geom.py`, `old_stats.py`, `fourier.py`, `aggregate.py`, `table.py`, `old_table.py`).
Raw outputs: `~/scratch/weekday/`. Summary: `results/weekday_calib_summary.json`. Figures: `results/weekday_calib_*.png`.
Compute: CPU only. Narval GPUs were in the POWEROUT20261027 reservation until 2026-11-03, so GPU job 7499750 was cancelled.
The run used about 2 h of login-node CPU for extraction and steering.

## Headline (v2, after the adversarial Opus review)
1. **The preregistered gate FAILS, but the gate was badly specified.**
   - The gate label was month y at mid layers. That label carries no answer signal there: class SNR is about 1e-4, and its f1 power is below the
     label-shuffle null. So the FAIL says little about the toolkit itself.
   - The 3×3 tuning grid does not change the outcome.
   - **What the data supports is power that depends on SNR:**

     | where | class SNR | what the class-level statistics do |
     |---|---|---|
     | z token | ≈ 1.8 | beat the SNR-matched line null: centroid turning 13/13, top-1 share 13/13, local-field rotation 13/13, distortion 9/13. P3/P4 as written essentially pass. |
     | last token, z labels | ≈ 0.01 | the same statistics mostly fail (turning 0–2/13, distortion 2–4/13) |

   - Two statistics fail everywhere:
     - TwoNN ID (P1) is misspecified: it measures the whole cloud, including k and template variation.
     - Distance cyclicity (cyc_G) also fails.
2. **Cyclic order is real (POST HOC statistics, correct null).**
   - Test statistic: share of the first circular harmonic (f1) in the reliable template-split centroid Gram. Null: an **order-relabel null** (class identities permuted on the Gram, 300 relabelings).
   - Result: p ≤ .007 on 12–13/13 mid layers and 7/7 late layers, for z labels at both positions. That is the resolution floor for 7 classes.
   - The **shape is not a closed 1-D circle.**
     - The cycle closes only partly: in the split-half Gram, the squared first–last distance is 1.3–3.2 times the mean adjacent one.
     - An approximate mixture decomposition (weekday z, last token) gives about 35% circle, 18% line and 48% simplex (categorical).
   - Centroid turning of 100–120° fits this f1 share. Turning is a high-pass statistic, so it adds no information beyond the Fourier share and is **not** separate evidence of "simplex".
3. **The old ot_geometry statistics were not calibrated by this study.**
   - O1 (flow-line turning, binary class pairs) gives the "flat" answer correctly here. Two classes that are translated copies of one shape have a linear log-ratio wherever they sit on a circle. So weekdays cannot be a positive control for O1.
   - O1-CV's 14/14 hits on y labels show it reacts to mixture heterogeneity, not to bending.
   - O3 geo_ratio is degenerate (1.000).
   - The O2 intermediate-class MMD (polyline < chord, 14/14 and 24/24) has no relabel null. Any off-chord intermediate gives this, including an unordered simplex, so it is suggestive only.
4. **What this means for the TruthfulQA "flat" verdict.**
   - That verdict answered a narrower question: curvature of a binary truth field, tested with O1 against an R-shift null. This study neither confirms nor refutes O1's power for that question.
   - The class SNR matches: TruthfulQA top-24 heads .004–.018 vs the last-token z labels here .005–.05, recomputed by the reviewer.
   - But sample size does not match: O1 fit sets are about 2.4k rows on TruthfulQA vs about 100 per pair here, and transfer of power across n is untested.
   - Safe statement: **"flat" is not positive evidence of flatness.** The toolkit's curvature statistics have no demonstrated power at SNR ≈ 0.01 with n ≈ 100. The binary formulation cannot express curvature through intermediate states.
   - A planted-curvature power curve (arcs embedded in real TruthfulQA residuals, matched SNR and n) is the needed next test.
5. **Steering: UNINFORMATIVE by design.**
   - Answer-label centroid deltas at L8–20 are about zero (SNR 1e-4), so the argmax never leaves the source and there is no Tuesday bump to compare.
   - No implementation bug was found (hook, KV-cache, t = 0 sanity check).
   - The informative version would steer at the z token with z-centroid deltas (SNR 1.8). Not run.

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
Reference values (7 classes): circle f1 1 / lin .33; equally spaced line .66 / 1.0; simplex .33 / .17. For 12 classes: line .63; simplex f1 .18.
f1 share alone cannot exclude a line. The order-relabel null (class identities permuted on the reliable Gram) is the test of cyclic order.

| run | mid f1 share | lin share | order-relabel p < .01 | f1 power ÷ label-shuffle max | class SNR (mid) |
|---|---|---|---|---|---|
| weekday z, last token | .62 | .37 | 12/13 | 8–112× | .011 |
| month z, last token | .42 | .23 | 13/13 | 10–80× | .010 |
| weekday z, z token | .53 | .28 | 12/13 | 65–220× | 1.8 |
| month z, z token | .39 | .19 | 13/13 | 40–352× | 1.8 |
| weekday y, late | .84 | .26 | 7/7 | 0.6–38× | .054 |
| month y, late | .71 | .23 | 6/7 | 2–207× | .30 |

Notes:
- y at mid layers also passes the order null (13/13), but there it is the circulant composition confound, not computation.
- Wrap closure is partial (reviewer): first–last d² / mean adjacent d² is 1.3–1.8 at the z token and 1.6–3.2 at the last token. adj = 1 is also met by an open arc or an ordinal horseshoe.

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

## What this means for paper W (v1 text, superseded where it conflicts with the v2 headline)
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

## Adversarial review (Opus) — objections and handling
| # | objection | severity | handling |
|---|---|---|---|
| 1 | Turning ≈108° is explained by the measured f1 share (high-pass); it is not evidence of "simplex" | fatal (framing) | Withdrawn. Categorical share comes from the mixture decomposition (≈35% circle / 18% line / 48% simplex). |
| 2 | The S8 label-shuffle null only shows class signal; the order needs an order-relabel null. The docstring line values were wrong | fatal (framing) | Order-relabel null added (`fourier.py`, p_order): p ≤ .007 on 12–13/13 layers. Reference rows corrected. |
| 3 | Cycle only partly closed; adj = 1 is also met by an arc or a horseshoe | major | "Ordinal arc + simplex, partially closed" (wrap 1.3–3.2). |
| 4 | "Toolkit fails" hides power at the z token (SNR 1.8); the gate label (month y) has no signal | major | Headline 1 rewritten: power depends on SNR; gate label invalid. |
| 5 | SNR comparison is fair (dimension irrelevant, reproduced .010–.017), but n differs about 20× | major | Headline 4 narrowed; planted-curvature power curve at matched n named as the next test. TruthfulQA SNR script `weekday/tqa_snr.py`. |
| 6 | O1 never had a positive control; binary-pair flatness is the correct answer here | major | Headline 3/4: the old verdict answers a different question; this study does not calibrate O1. |
| 7 | O2 intermediate-MMD is near-circular (the polyline passes through the fit centroid) | major | Demoted to suggestive; needs a relabel and line2 null. |
| 8 | Label the late window and the SNR comparison as post hoc; A5 does not drive the FAIL | minor | Done. |
| 9 | Number ranges (8–200×) | minor | Replaced with the per-run table. |
| 10 | Steering design could not work (y deltas at SNR 1e-4) | minor | "Uninformative by design"; z-token steering named. |
| 11 | Weekdays are a weak positive control; plant curvature instead | major | Adopted as the next step. |

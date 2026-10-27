# Weekday/month calibration of the geometry toolkit — Gemma-2-2B-it (PARTIAL, pre-review)

Status: **partial** — weekday last-token done; months, z-token control and steering running. Prereg: `results/weekday_calib_prereg.md` (commit 910da50).
Code: `weekday/`. Raw: `~/scratch/weekday/`. CPU only (Narval GPU partition is in the POWEROUT20261027 reservation until 2026-11-03).

## Amendments (all after the prereg commit)
- A1 (technical, before data): prefix "The answer is **" — Gemma emits " **" after "The answer is". Qwen weights incomplete → not run.
- A2 (synthetic, before data): mean turning of any closed polygon is ≥ 360/C (a back-and-forth line also gives 51°) → read the **median** corrected angle.
- A3 (bug fix, after data): random within-class split halves have complementary (z,k) composition → negative cross-Gram eigenvalues. Split by template (3/6).
- A4 (POST HOC): S7 cyclic order in the split-half top-2 subspace (adj, wind).
- A5 (POST HOC): SNR-matched "unrolled line" null line2 (equally spaced on one direction, same centroid spread, permuted real residuals).
- A6 (POST HOC): z-token position (the token naming the start day) as a higher-SNR positive control.

## Weekday, last prompt token (546 prompts; accuracy .28: k=1,2,3 → 1.00/1.00/.79, k ≥ 4 → 0–.21)
Middle layers 6–18; "x/13" = layers where data beat every null rep.

| statistic | data (median mid) | flat-G | line | line2 | shuffle | verdict |
|---|---|---|---|---|---|---|
| TwoNN ID (P1: ≤ 3) | 6.9 | lower 10/13 | – | – | – | **P1 fails** (ID ≈ 7, not 1–2) |
| distortion (P3) | .14 | above 13/13 | 1/13 | 2/13 | – | **P3 fails**: comes from the label-free point shape (line null equal) |
| cyclicity d_G (P2) | .28 | – | 4/13 | 4/13 | 4/13 | **P2 fails** (≥ 2/3 needed) |
| cyclicity d_E | −.18 | – | 0/13 | 1/13 | 0/13 | no power |
| corrected turning, median (P4) | 99° | – | 0/13 | 0/13 | – | **P4 fails**: ≈ noise/simplex level |
| top-1 split-half share (P4b < .8) | .49 | – | 11/13 | 13/13 | – | passes |
| local-field rotation | 69° | – | – | – | 13/13 below | coherent, but no line null beats it |
| **S7 adj (post hoc)** | **1.0** | – | 9/13 | 10/13 | 11/13 | **detects** |

Labels by the start day z (cleaner, no composition confound): only cyc_E vs line2 (11/13) and top-1 (11/13) reach 2/3; adj .71 (7/13).
Labels by model prediction: no cyclic structure (adj .14). Correct-only subset (n = 154): nothing detects.

## Old statistics from ot_geometry.md applied to weekday class pairs (i→i+2, i→i+3; layers 4/8/12/16/20; z labels and y labels)
- **O1 flow-line turning**: α = 100 → 0.0° on every pair (the MLP is linear); CV-selected α → 0–33°, always within the G-shared / R-shift null
  range (e.g. 31.0° vs nulls 29.6 / 30.4°). **0 detections** — the old statistic is blind to this known cyclic structure.
- **O3 geo_ratio** (two-class graph, as in ot_geom): 1.000 on every pair; all-class graph 1.0–1.65 on a few. OT R²_s50 = 1.00 for data and the R-shift null. **No power.**
- **O2 target MMD**: d* ≈ 0.55–0.95|Δμ| — indistinguishable in form from the TruthfulQA "overshoot" pattern. The new readout, MMD to the
  intermediate class, is lower along the centroid polyline than along the straight chord in every pair (z labels, L20: .08 vs .04).
  This is the only old-pipeline readout that sees the bend, and only once the intermediate classes are used.

(Full tables, months and steering: to follow.)

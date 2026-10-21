# Is the NEG→POS transport flat? OT geometry of IDEEA's head activations (Gemma-2-2B, TruthfulQA)

**Status: PARTIAL. Jobs queued (CPU array 6593005, power 6593029); numbers below are setup and smoke tests only.**
Prereg: `results/ot_geometry_prereg.md` (+ amendment A1). Code: `ot_geometry/`.

## Setup (done)
- Data = the IDEEA cache `head.pkl` restricted to the development split (395 questions, 3049 rows: 1415 POS, 1634 NEG).
  Fit = IDEEA `train` questions (2427 rows); held-out = IDEEA `val` questions (622 rows, 79 q). Test questions were never loaded.
- Heads = IDEEA top-24 (ITI val-accuracy ranking, paper K=24) + 8 random controls (`ot_geometry/heads.json`).
- No GPU used (activations were cached).

## Early methodological finding (synthetic, before real data)
The raw flatness statistic R² = ||mean d||²/mean||d||² **cannot detect curvature in 256-d at n≈10³**: finite-sample
matching noise (≈ 2·dim per point) dominates the field. A 2-mode field with true R² = 0.5 gave 0.017 vs flat-null 0.019.
A kNN-smoothed field (average d over 50 NEG neighbours) does detect it (0.35 vs null min 0.48) and stays inside the
null for flat data. The primary statistic was switched to R²_s50 before any real-data run (amendment A1).

## Smoke test (L13H5, pw8 space, 2 null reps; not a result)
R²_s50 0.64 vs translation nulls ≈ 0.93 and the affine null 0.70; kNN-local predictor cos 0.78 vs mean-diff 0.58.

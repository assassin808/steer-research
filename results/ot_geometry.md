# Is steering space flat? Trait flows (and OT as a baseline) on IDEEA's head activations (Gemma-2-2B, TruthfulQA)

**Status: PARTIAL. CPU arrays queued: trait flow 6595081 (primary), OT baseline 6595082, power 6593029.**
Prereg: `results/ot_geometry_prereg.md` (A1 power amendment, A2 trait-flow reframe, deviation D1). Code: `ot_geometry/`.

## Setup
- Data = IDEEA cache `head.pkl`, development split only (395 q, 3049 rows: 1415 POS, 1634 NEG). Fit = IDEEA `train`
  questions (2427 rows); held-out = IDEEA `val` questions (622 rows, 79 q). Test questions never loaded. No GPU used.
- Heads = IDEEA top-24 (ITI val-accuracy ranking, paper K = 24) + 8 random controls (`ot_geometry/heads.json`).
- Primary object (A2): trait score s = log-ratio classifier (LR; tanh MLP), flow V = P_T ∇s / |P_T ∇s|.
  OT barycentric displacement is now only a comparison column.

## Early methodological finding (synthetic, before real data)
The raw OT flatness R² cannot detect curvature in 256-d at n ≈ 10³ (true field R² 0.5 → 0.017 vs flat-null 0.019);
a kNN-smoothed field can (0.35 vs null min 0.48). Primary OT statistic switched to R²_s50 before any real run.

## Post-hoc exploratory: the de-facto question pairing (not preregistered)
POS and NEG rows of one question share its context. Per-question displacement δ_q = mean POS_q − mean NEG_q
(z space, 355 questions with ≥ 2 of each), split-half cross-product to remove within-question noise:
share of the field's second moment that is the constant shift, R²_pair = (mean δ^A)·(mean δ^B) / mean(δ^A_q·δ^B_q):
- top-24 heads: median ≈ 0.13 (range 0.05–0.18); mean cos(δ_q, Δμ) 0.15–0.30;
- random heads: 0.01–0.10.
So per question, the "correct minus incorrect" shift is mostly NOT the global mean difference. Caveat: δ_q also
contains answer-content differences that are not "truth" (the correct and incorrect reference answers say different
things), so this bounds the constant share from below rather than measuring curvature of a trait field.

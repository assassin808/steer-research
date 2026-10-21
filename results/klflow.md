# KL gradient flow vs frozen velocity (P2), conditional OT (P3), guidance steering (P1, gated)

**Status: PARTIAL.** Jobs queued:
- 6691863: P2, 32 heads × {z with 20+20 null reps, pw32 data-only};
- 6691864: P3.

Prereg: `results/klflow_prereg.md` (+ amendment A1: KDE bandwidth grid extended downward after a smoke test).
Code is in `klflow/`; raw outputs are in `~/scratch/klflow_out/`. CPU only so far.
Note: written in `~/scratch/steer_live` because of the /home Lustre outage.

Smoke test (L13H5, z, 5 steps only; not a result):
- KDE-CV bandwidth hit the grid edge (c = 0.5).
- Over the first 0.5|Δμ|, straight Δμ lowers MMD² the fastest (0.0152 → 0.0044; floor 0.0034).

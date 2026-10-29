# Planted-curvature power curves for the geometry statistics

Status: **PARTIAL (v0, 2026-10-29 04:40)**. Done: centroid grid (A) and point grid (P). Still running: practical-null check, binary O1/OT grid (B).
No adversarial review yet.
Prereg: `results/power_curve_prereg.md` (commit bc8463b). Code: `power/`. Raw outputs: `~/scratch/power/`.

## Done so far
- **Part A** (centroid statistics turn / offline / quad⊥ / ord1 / f1c): 280 cells × 400 simulations.
  Backgrounds tqa ×3 heads, wk, wkz, syc, opt; planes random/top; L = 5, 7; n = 20–1000; iid/q modes.
- **Part P** (frot, fincoh, dist): 84 cells × 40 simulations.
- Real-data SNR placement (`~/scratch/power/place.json`):

  | data | L | n per class | SNR |
  |---|---|---|---|
  | TruthfulQA top-24 | 2 | 1128 / 1299 | median .0098 (.004–.019), question share of W .52 |
  | weekday last token (L12) | 7 | 78 | .014 |
  | weekday z token | 7 | 54 | 1.86 |
  | PoC v3.1 syc L14, pos vs neg | 2 | 300 | .036 (question share .67) |
  | PoC v3.1 opt L14, pos vs neg | 2 | 300 | .045 (question share .69) |

## First readings (oracle power; practical checks pending)
1. **quad⊥ is the most powerful curvature statistic.** At 80% power for arc 90°, L = 7, TQA background:
   - quad⊥ needs SNR ≈ 2.2/n;
   - offline needs ≈ 6.6/n;
   - the kit's turning angle needs ≈ 200/n, about 100× worse.

   n·SNR ≈ constant holds for n ≥ 50 (P3 PASS).
2. **Turning (kit S6) is nearly useless and misses bends.** On bend90 its power stays ≤ .08 at every SNR, because the median over vertices ignores a single bend.
   - P2 (weekday consistency) passes for turn: power 0 at the last token, 1.0 at the z token, reproducing the real weekday result.
   - At the weekday last-token SNR, quad⊥, offline and f1c would have detected a planted 309° arc (power .8–1.0).
3. **Question structure helps a lot when the same questions appear at every level.**
   - In a balanced design, question means cancel exactly in centred centroids.
   - With persona data (question share ≈ .67), the q design needs ≈ 3× lower SNR than iid.
4. **Oracle false-positive issues (to be settled by the practical null):**
   - Under the arc 0 threshold, the straight line with saturating spacing (`sat`) gives:
     - turn FPR .115 (median);
     - quad⊥ .14 (max .38);
     - offline .05 (OK).
   - The local-field statistics (frot, fincoh) fire on `sat` at high SNR.
   - First practical-null result: frot against the line null has FPR 0 on arc 0 and on `sat`, and power 1 on arc 90 at SNR 3. The projected-position line null absorbs unequal spacing.
5. **P4 as written FAILS.** Centroid statistics do respond to `clust` plants (offline up to 1.0, turn up to .93): the per-level cluster fractions fluctuate, so the sampled centroids leave the line. Whether this survives the practical line null is pending.

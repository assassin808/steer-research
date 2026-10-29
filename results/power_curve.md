# Planted-curvature power curves for the geometry statistics

Status: **v2, revised after the adversarial Opus review** (review table at the end).
- Prereg: `results/power_curve_prereg.md` (commit bc8463b). Everything marked **POST HOC** was added after the review.
- Code: `power/`. Raw outputs: `~/scratch/power/`.
  - Grids: `partA.py` (centroid), `partP.py` (point), `partB.py` (binary).
  - Checks: `nullcheck.py` (practical nulls), `designcheck.py` (persona design check, POST HOC).
  - Summaries: `agg.py`, `tables.py`, `preds.py`, `design.py`, `place.py`.
- Compute: login node only. The POWEROUT20261027 reservation covers every Narval compute node, and the login node allows about 4–5 cores per user.
- Figures: `results/power_curve_heat_arc60.png`, `..._arc90.png`. Each shows oracle power over (n × SNR) for L = 7, with real-data points marked.
- Tables: `results/power_curve_tableA.md` (minimum SNR for 80% power, centroid statistics), `results/power_curve_tableP.md` (local-field rotation).

## Headline (v2)
1. **TruthfulQA: the "flat" verdict has no power.** Two separate reasons:
   - **Curvature through intermediate states is unidentifiable from a binary label.** At L = 2, two centroids always lie on a line, so every
     statistic has power equal to its FPR on planted arcs.
   - **O1, the statistic behind the verdict, cannot see IDEEA-like heterogeneity at all, by construction.** If the shift direction depends on a
     latent cluster, the flow inside each cluster is still straight. At the TQA cell (n = 1200 per class), O1 turning is ≈ 0.001° for both the
     straight plant and the 90° cluster plant, at SNR .01 and .1. O1 reacts only to 180° (opposite-direction) heterogeneity: about 20°, power 1.
   - Other statistics do react to heterogeneity, but neither is a calibrated test on real TQA:
     - 1 − OT R²_s50: clust90 power .33 at SNR .01 and .94 at SNR .1 (oracle, random plane, 18 simulations).
     - local-field coherence (fincoh): see §5.
     - On real TQA, fincoh rejects the R-shift null by about 18 SD. The probable cause is question structure, so a within-question null is needed first.
2. **For graded designs (L ≥ 5), use ord1 as the order gate and quad⊥ as the curvature test.**
   - **ord1**: order share along the linear contrast, with an exact order-relabel test.
   - **quad⊥**: the quadratic-in-level centroid component orthogonal to the line direction (split-half), with a line-null permutation test.
   - Power ranking, oracle (arc 90°, L = 7, TQA background), n·SNR needed for 80%:
     - quad⊥ ≈ 2.2;
     - offline ≈ 6.6;
     - the kit's turning angle ≈ 200;
     - ord1 (order only) ≈ 0.2–0.4.
   - n·SNR ≈ constant for n ≥ 50 (P3 PASS).
   - The turning angle is blind to a single bend.
   - Distortion has power ≤ .30 even at SNR 3.
3. **Calibration is not automatic.** quad⊥ and offline need care:
   - **Plain line null, no question×level cell effect.** FPR ≤ .075 at n = 300, but .06–.10 at n = 50 (P1 FAILS as written for offline/quad⊥ at n = 50). Testing at **p ≤ .02** brings every checked cell to ≤ .06.
   - **With a question × level cell effect** (in the persona data, its measured size is ≈ 0.55× the 5-sample cell-mean noise), the prereg's within-question **row**-permutation null breaks: FPR .15–.92.
   - A **cell-permutation null** (whole (question, level) cells permuted across levels within question) restores FPR to .03–.13 (mostly ≤ .10).
   - **Any persona-trait analysis must use the cell-permutation null.**
4. **Persona study size** (POST HOC design simulation, `designcheck.py`).
   - Inputs:
     - SNR measured from PoC v3.1 pos/neg at L14 (syc .036, opt .045);
     - a 5-level ladder spanning those endpoints: SNR_L ≈ .018 / .023, also tested at ×0.5;
     - cell effect κ = .55;
     - cell-permutation null, quad⊥ at p ≤ .05.
   - Results:

     | design | rows | syc arc60 (random / Δμ plane) | opt arc60 (random / Δμ plane) |
     |---|---|---|---|
     | 35 q × 5 levels × 5 samples | 875 | ×1: .77 / .92; ×0.5: .30 / .57 | ×1: 1.0 / .98; ×0.5: .60 / .68 |
     | 70 q × 5 levels × 3 samples | 1050 | ×1: .98 / .98; ×0.5: .50 / .73 | – |

   - **Cell effects average out over questions, not over samples. Spend the budget on questions.**
   - **Recommendation:** ≈ 70 questions × 5 levels × 3 samples. That gives ≈ 98% power for a 60° arc at the endpoint-implied SNR, and 50–75% if the ladder reaches only half that SNR.
   - The v3.1 pool has 60 questions per trait, so ≈ 10–20 more are needed.
   - Read activations **downstream** of the steering layer, or use prompted ladders. At the steering layer, a fixed-vector intervention is straight by construction.

## 1. Setup (preregistered unless marked)
- **Backgrounds.** Real class means are removed; each background is stored in full-rank PCA coordinates (an isometry).

  | id | rows | participation ratio | question share of W |
  |---|---|---|---|
  | tqa L13H5 / L21H6 / L15H2 (z, 256-d) | 3049 | 25 / 76 / 69 | .53 / .42 / .48 |
  | wk (weekday last token, L12) | 546 | 4.5 | .98 (78 k × template groups) |
  | wkz (z token, L12, deduplicated) | 378 | 4.5 | .90 |
  | syc / opt (v3.1 w23 pos+neg, L14) | 600 | 35 / 21 | .67 / .69 |
- **Planting**: arcs θ ∈ {0, 30, 60, 90, 180, 309, 360}, `sat` (straight line, saturating spacing), bend 60/90, clust 60/90/180.
  Plane: a random 2-plane in the row span (primary), the top-2-PC plane (sensitivity), and POST HOC a plane containing the real Δμ.
  SNR = (1/L)·Σ‖ā_ℓ − ā‖²/W.
- **Oracle power**: the threshold is the 95% quantile of arc-0 simulations in the same cell, with common random numbers.
  The order statistics (ord1, f1c) use their exact order-relabel p.
  - FPR at SNR = 0: ord1 .050 (mean over cells), f1c .025.
- **Deviations (budget):**
  - P grid at 40 simulations (prereg 60).
  - Distortion only at n ≤ 50, on 3 shapes.
  - B grid cut to tqa, n = 1200, SNR {.01, .1}, arc0/clust90/clust180, 18 simulations.
    - The prereg grid was 2 backgrounds × 4 n × 6 SNR × 4 shapes × 40 simulations.
    - One O1 evaluation at n = 1200 takes about 4 CPU-minutes.
  - quad⊥ is normalised by the pooled ‖b1‖². The prereg wrote ⟨b1^A, b1^B⟩.
  - Seeds come from Python `hash()` and are not reproducible across interpreter runs (designcheck uses crc32).

## 2. Validity (P1): practical tests
**Line null** (prereg): centroids projected on the top PC of the observed centroids, keeping their projected positions; residuals permuted
across levels (within question in q mode); 99 reps; 100 simulations per cell. Rejection rate at p ≤ .05, and in brackets at p ≤ .02.

| cell (L = 7) | stat | arc0 FPR, SNR .03/.3/3 | sat FPR | arc90 power | clust90 |
|---|---|---|---|---|---|
| tqa iid n50 | turn | .07/.08/.08 (.03/.01/.01) | .07/.02/.03 | .03/.06/.65 | .03/.01/.02 |
| | offline | .10/.06/.07 (.04/.01/.02) | .11/.08/.07 | .28/1/1 | **.16**/.10/.11 |
| | quad⊥ | .09/.08/.07 (.05/.05/.06) | .08/.06/.10 | .44/1/1 (.26/1/1) | .08/.12/.10 |
| tqa iid n300 | turn | .02/.02/.05 | .08/.07/.08 | .06/.32/1 | .05/.01/.02 |
| | offline | .04/.06/.07 (.02/.02/.02) | .05/.05/.07 | 1/1/1 | .09/.12/.08 |
| | quad⊥ | .03/.06/.05 (.02/.02/.03) | .05/.03/.06 | 1/1/1 | .05/.05/.07 |
| syc q n50 | offline | .10/.06/.06 (.05/.03/.03) | .08/.08/.06 | .86/1/1 | .08/.03/.03 |
| | quad⊥ | .09/.05/.06 (.01/.01/.03) | .04/.03/.03 | 1/1/1 | .05/.06/.04 |
| syc q n300 | offline | .04/.08/.07 (.01/.02/.03) | .07/.05/.03 | 1/1/1 | .03/.02/.01 |
| | quad⊥ | .03/.04/.05 (.01/.01/.01) | .06/.06/.05 | 1/1/1 | .05/.06/.07 |
| tqa iid n50 (40 sims, 39 reps; SNR .3/3) | frot | .02/.02 | .02/.02 | .02/1.0 | 0/0 |

Reading:
- **P1 as written FAILS for offline and quad⊥ at n = 50** (FPR up to .10 and .09; reviewer replication with 210 simulations: .086 and .071). P1 passes at n = 300. turn and frot pass.
- **The `sat` false positives of the oracle are a threshold artefact.** Under the oracle, sat FPR is .12–.24 for turn and quad⊥. The practical null keeps the projected spacing, which brings sat FPR to .02–.11.
- **P4 FAILS.**
  - The oracle "detects" clust with centroid statistics: offline up to .99, turn up to .93. Cluster fractions jitter between level samples, which adds off-line centroid noise that the arc-0 null lacks.
  - The practical null absorbs most of this, but not all: offline .16 at tqa n50.
  - offline (eigenvalues clipped at 0) reacts to any extra noise that differs between levels.
- **TRIED, NOT ADOPTED: "perp null"** (permute only the component orthogonal to the line). It does not fix the n = 50 excess. Reviewer (210 simulations): offline .081, quad⊥ .076.
- **POST HOC fix used for claims:** test at p ≤ .02. Every checked cell is then ≤ .06; power at SNR ≥ .3 is unchanged (1.0).
- **Cell effects** (`designcheck.py`, POST HOC; syc/opt, L = 5, Q = 35, r = 5, κ = .55; 60 simulations, 39 reps):

  | null | quad⊥ arc0 FPR | offline arc0 FPR |
  |---|---|---|
  | row permutation (prereg-style) | .15–.65 | .32–.92 |
  | cell permutation | .05–.10 | .03–.13 |

  Without a cell effect, both nulls give .03–.13. The q mode of the main grid has no cell effect, so its "q design is 3× cheaper" figure is optimistic. The design simulation above replaces it for sizing.

## 3. Minimum SNR for 80% power (oracle, random plane; n per level = 20 / 50 / 100 / 300 / 1000)
| bg, mode | L | shape | turn | offline | **quad⊥** | ord1 (order, straight plant) |
|---|---|---|---|---|---|---|
| tqa iid | 7 | arc60 | >3 / >3 / >3 / 1.8 / .59 | .71 / .23 / .13 / .059 / .010 | **.24 / .13 / .068 / .021 / .0068** | .0075 / .0029 / .0017 / ≤.001 / ≤.001 |
| tqa iid | 7 | arc90 | >3 / >3 / 2.0 / .71 / .22 | .23 / .09 / .066 / .021 / .0066 | **.14 / .067 / .022 / .0074 / .0022** | same |
| tqa q | 7 | arc60 | >3 / >3 / 2.2 / .73 / .23 | .27 / .13 / .071 / .022 / .0071 | **.14 / .073 / .026 / .0079 / .0025** | .0045 / .0018 / ≤.001 … |
| syc q | 5 | arc60 | 2.1 / .73 / .45 / .15 / .053 | .22 / .073 / .036 / .013 / .0038 | **.083 / .061 / .021 / .0074 / .0022** | .0034 / .0015 / ≤.001 … |
| syc q | 7 | arc90 | 2.4 / .85 / .68 / .22 / .068 | .073 / .027 / .019 / .0065 / .0018 | **.050 / .018 / .0075 / .0023 / ≤.001** | .0022 / ≤.001 … |
| syc iid | 7 | arc60 | >3 / >3 / >3 / 1.1 / .33 | .61 / .20 / .081 / .025 / .008 | **.22 / .076 / .063 / .021 / .0059** | .0068 / .0022 / .0015 … |
| wk iid | 7 | arc60 | >3 / >3 / >3 / 2.8 / 1.1 | 1.9 / .67 / .24 / .076 / .026 | **.72 / .23 / .18 / .062 / .013** | .019 / .007 / .0035 / .0011 / ≤.001 |

Full table: `power_curve_tableA.md`.

What modulates power:
- **Effective dimension.** wk (participation ratio 4.5, a few massive Gemma dimensions) needs 2–3× more SNR.
- **Plane.** A curve in the top-2-PC plane needs 2–3× more SNR than one in a random plane. Persona Δμ-aligned planes are *not* harder for quad⊥ (design check).
- **Shape.** quad⊥ and offline see bend90 at SNR .01–.03 (n = 100). turn is blind to it (median over vertices).
- **Prediction status.**
  - P3 PASS. P6 (quad⊥ ≥ turn) PASS.
  - P5 (order cheaper than curvature) PASS, but this is a design rationale, not a finding.
  - turn returns NaN, counted as "not detected", at very low SNR. Its oracle power at SNR ≤ 3e-3 is therefore below α.

## 4. Weekday consistency (P2) — PASS
Planted arc 309°, L = 7:

| | turn | offline | quad⊥ | f1c (order) |
|---|---|---|---|---|
| wk, n = 78, SNR .014 (real last token) | **.00** | .81 | 1.00 | .96 |
| wkz, n = 54, SNR 1.86 (real z token) | **1.00** | 1.00 | 1.00 | 1.00 |

- The benchmark reproduces the kit's real outcome: turning fails at the last token and succeeds at the z token.
- It also matches the order-statistic success at the last token.
- Qualitative only: the real cycle is ≈ 35% circle, and the turn "0" is partly NaN.

## 5. Binary designs (L = 2) and TruthfulQA
- **Arcs at L = 2: unidentifiable.** frot/fincoh arc90 power .02–.12 (= FPR).
- **O1 at the TQA cell** (n = 1200, 18 simulations, random plane):
  - arc0 0.0006°;
  - clust90 0.0007° at SNR .01 and 0.0003° at SNR .1 (oracle power .11 / .00; the .11 is noise at the 0.001° scale);
  - clust180 16–25° (power 1.0).
  - In real TQA the R-shift null itself turns 14° (non-Gaussian shape along Δμ), which would swamp any clust90 effect. **O1 has no power against IDEEA-like 90° heterogeneity.**
- **1 − OT R²_s50**: clust90 power .33 (SNR .01), .94 (SNR .1); clust180 .94 / 1.0. Oracle, random plane, 18 simulations.
- **fincoh**:
  - Oracle clust90 power at SNR .01, random plane: .38 / .98 / 1.0 for n = 20 / 100 / 300.
  - With a Δμ-aligned or top-PC plane (reviewer), power at n = 300 falls to .60 / .50 and is 1.0 only at n = 1200.
  - On real TQA L13H5, fincoh is .620 vs .392 under the R-shift null. This is probably question structure (question share .52: kNN fields link same-question answers), not heterogeneity. It needs a within-question null before any claim.
- **Decision (prereg rule):** O1 clust90 power < .8 → rename the ot_geometry "flat" verdict to **"underpowered (no power)"**. Stated more strongly: O1 is blind to this alternative by construction.

## 6. Real data on the maps
| data | design | SNR | placement |
|---|---|---|---|
| TruthfulQA top-24 heads | L = 2, n ≈ 1200/class, 395 q | median .0098 (.004–.019) | curvature unidentifiable; O1 blind to clust90. A 5–7-level ladder at the same endpoint SNR (SNR_L ≈ .0045) would need n ≈ 460–740 per level (quad⊥, q, no cell effect) |
| weekday last token | L = 7, n = 78 | .014 | turn 0; quad⊥ / f1c ≈ 1 |
| weekday z token | L = 7, n = 54 | 1.86 | all ≈ 1 |
| PoC v3.1 syc / opt (pos vs neg, L14) | L = 2, n = 300, 60 q | .036 / .045 | 5-level ladder ≈ .018 / .023 → see headline 4 |
| PoC v3.1 W1 design | L = 6, 30 q × 20 | ≈ .017–.021 | only 30 questions: with cell effects, power is about that of 35 q × 5 (≈ .77–1.0 at ×1 SNR); more samples per question do not help |

## Decisions
1. **TruthfulQA:** the verdict is "no power": no power for curvature (binary label), and O1 has no power for IDEEA-like heterogeneity.
   Next test, if wanted: fincoh or OT with a within-question null and a Δμ-aligned planted control.
2. **Persona statistics:**
   - ord1 (gate: is there graded order?);
   - then quad⊥ against the **cell-permutation** line null at p ≤ .02–.025.
   - offline as a secondary check; it is the more robust statistic to unequal spacing under the oracle, but the less powerful one.
   - Do not use turning, frot or distortion for claims.
3. **Persona design:** ≈ 70 questions × 5 levels × 3 samples per (question, level). Add questions beyond the 60-question v3.1 pools.

## Limitations
- Planted shapes are pure (one curve, one plane). Real structure is mixed.
- Backgrounds are resampled with replacement when n·L exceeds the rows. That is a bootstrap world, optimistic for kNN statistics.
- The persona SNR comes from prompted endpoints at L14. A steered or prompted ladder may not reach it, hence the ×0.5 column.
- The cell-effect size (κ ≈ .55) is the reviewer's estimate from pos/neg Δ_q spread. Measure it in the pilot.
- The B grid is tiny (18 simulations, random plane only).

## Adversarial review (Opus) — objections and handling
| # | objection | severity | handling |
|---|---|---|---|
| 1 | P1 fails for offline/quad⊥ (FPR .07–.11 at n = 50); oracle "stands in" breaks the prereg rule | major | P1 reported as FAIL at n = 50. POST HOC: claims use p ≤ .02 (≤ .06 everywhere checked). |
| 2 | The perp-null amendment does not work; its docstring misquoted FPR | major | Not adopted, docstring corrected, logged as TRIED. |
| 3 | P4 misreported (practical clust90 offline .16) | major | P4 reported as FAIL, with the mechanism. |
| 4 | O1 cannot see clust by construction; "underpowered" understates it | major | Headline 1 and §5 say "blind by construction". |
| 5 | Random-plane world is easier than real data for point/flow statistics (fincoh power .95 → .60 in a Δμ plane) | major | fincoh claims restated with plane dependence. Persona design re-simulated in a Δμ plane (quad⊥ holds). |
| 6 | fincoh rejects the R-shift null on real TQA (.62 vs .39), likely question structure | major | Stated. No heterogeneity claim; a within-question null is required. |
| 7 | q mode ignores question × level cell effects; the row-permutation null is not exchangeable under them | major | POST HOC `designcheck.py`. Row null FPR .15–.92 with κ = .55; cell-permutation null fixes it. Design recommendation changed to more questions. |
| 8 | Persona design untested at L = 5 with the practical null; interpolation is coarse; the SNR_L assumption is untested | major | Exact design simulated (Q = 35 vs 70, two planes, ×0.5/×1 SNR, cell effect, cell null). |
| 9 | B too thin for a B2 verdict | minor | B2 reported as partial power, with the number of simulations. |
| 10 | quad⊥ denominator differs from the prereg; oracle sat-FPR | minor | Listed under deviations; sat handled by the practical null. |
| 11 | turn NaN → below-α oracle power; P2 easy to pass | minor | Stated in §3/§4. |
| 12 | Misquote: .35 vs .40 | minor | Removed. |
| 13 | `hash()` seeds not reproducible | minor | Listed under deviations; designcheck uses crc32. |
| 14 | P5 is uninformative | minor | Reframed as design rationale. |

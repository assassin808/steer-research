# KL gradient flow vs frozen velocity (P2), conditional OT (P3), guidance steering (P1, gated)

Status: **v1, before adversarial review.**

- Prereg: `results/klflow_prereg.md`. It has two amendments:
  - A1: KDE bandwidth grid extended downward, after a 5-step smoke test.
  - A2: POST HOC measurement fixes after the first P2 run (see "Deviations").
- Code: `klflow/`. Summaries: `results/klflow_summary.json`, `results/klflow_p3.json`. Raw outputs: `~/scratch/klflow_out/`.
- CPU only. **P1 (GPU) was not run, because the prereg gate failed.**
- Data, splits and heads are the same as `ot_geometry`:
  - Gemma-2-2B TruthfulQA, IDEEA head activations, development split only;
  - fit = IDEEA train questions; held-out = 79 val questions (335 NEG, 287 POS);
  - heads: IDEEA top-24 + 8 random;
  - z space is primary, pw32 is a sensitivity check.

## Headline
1. **P2a fails.** The frozen-velocity paths (straight Δμ, straight probe w, frozen ∇s) do pass through an MMD minimum
   and then diverge (24/24 heads), as predicted. But the guidance flow ∇s + γ∇log p+ does not plateau: 0/24 heads.
   The particle KL flow plateaus on 0/24 heads at h_smooth and 4/24 at h_cv.
2. **P2b fails.** In z, the best guidance or particle path reaches a lower minimum MMD than straight Δμ on only 21% of
   heads at the primary bandwidth. Its gap beats the null 97.5% quantile on only 0–12% of heads (both nulls and both bandwidth variants).
   At bandwidths ×0.5 and ×2 it is 12% and 58%; the threshold was 70%.
3. **P1 gate: not passed.** The GPU arm was not run.
4. **P3 fails.** Within-question (conditional) OT aligns with ∇s *worse* than unconditional OT on 24/24 heads
   (.095 vs .129). It is still above a shuffle null (.067).
5. **What does hold.**
   - Guidance and particle paths stay inside the data support (100% of points), while straight and frozen paths leave it
     after about 6|Δμ|.
   - The particle flow does not overshoot the trait score: final s ≈ 1.3 vs 9.7 for frozen paths.
   - But guidance stays in support by *drifting toward high-density POS regions*, not by matching the POS
     distribution: its MMD rises again.
   - Straight Δμ remains the best simple transport in z.

## P2 — moving the held-out NEG cloud toward POS
**Metric (A2).** MMD² between moved NEG rows of question-half A and POS rows of the other half B, averaged over both
directions. The RBF bandwidth is ×{0.5, 1, 2} × the median pairwise distance.
- "min/start" is the minimum MMD² along the path divided by the starting value.
- "÷floor" divides the minimum by the floor, the MMD² of an independent fit-POS sample to the same target.
- d* is the median displacement at the minimum, in units of |Δμ|.
- "plateau" means rise ≤ 0.1 after the minimum; "overshoot" means rise ≥ 0.5.

z, top-24 heads, primary bandwidth (medians):

| path | min/start | min ÷ floor | d* | plateau | overshoot | in support at end | final s |
|---|---|---|---|---|---|---|---|
| straight Δμ | **.56** | 2.36 | 0.85 | 0/24 | 24/24 | 0% | 9.7 |
| straight probe w | .84 | 3.38 | 0.50 | 0 | 24/24 | 0% | – |
| frozen ∇s (unit speed) | .85 | 3.36 | 0.50 | 0 | 24/24 | 0% | 9.7 |
| guidance KDE k = 1, h_smooth | .86 | 3.35 | 0.59 | 0 | 24/24 | 100% | 7.3 |
| guidance KDE k = 1, h_cv | .88 | 3.40 | 0.59 | 0 | 24/24 | – | – |
| guidance DSM k = 1, σ_smooth | .87 | 3.38 | 0.50 | 0 | 24/24 | – | – |
| particle KL flow, h_smooth | .60 | 2.48 | 0.82 | 0 | 12/24 | 100% | 1.3 |
| particle KL flow, h_cv | .999 | 3.87 | 0.05 | 4/24 | 18/24 | – | – |

**P2b**: best of {guidance K/D at all k, particle} vs straight Δμ.

| bandwidth | heads where it is lower | gap > R-shift null 97.5% (h_smooth / h_cv) | gap > G-shared null 97.5% (h_smooth / h_cv) |
|---|---|---|---|
| ×0.5 | 12% | 12% / 21% | 17% / 67% |
| ×1 | 21% | 8% / 0% | 12% / 12% |
| ×2 | 58% | 12% / 0% | 8% / 8% |

Note on the h_cv columns: "gap > null quantile" can hold even when the data gap is negative. At ×0.5 under G-shared,
the h_cv gap is less negative than in the null, but Δμ still wins (lower on 0% of heads). That is not an advantage.
The median data gap is −0.0010 at ×1, which favours Δμ. The null-gap medians are ≈ +0.0001 to +0.0004.

**Nulls (20 reps each).**
- Under both nulls, straight Δμ reaches min/start .03 and the particle flow (h_smooth) .005–.011. Both get essentially
  all the way to the target.
- On data, both stall at about 2.4× the floor.
- So the NEG→POS difference is not a translation, which agrees with the OT column of ot_geometry, and the KDE particle
  flow does not close it in 256-d either.
- Under the nulls, guidance KDE also overshoots on every rep (min/start .49–.77), so its failure to plateau is a
  property of the method, not of the data.

**pw32 sensitivity** (data only, no nulls):
- The particle flow at h_smooth plateaus on 75% of heads (overshoot on 8%).
- It reaches a lower minimum than straight Δμ in floor units: 1.89 vs 2.15 at ×1, 1.99 vs 2.34 at ×0.5, 1.90 vs 2.07 at ×2.
- Its raw minimum MMD beats straight Δμ on 42–71% of heads, depending on bandwidth.
- This is the only setting where the KL-flow picture partly appears. It was not preregistered as primary and has no
  nulls, so it is suggestive only.

**Random-8 heads (z).** Same pattern, weaker signal: straight Δμ min/start .89, particle .93.

**Interpretation.**
- The frozen velocity overshoots, which is the mechanism the PI predicted. Straight paths cross the support boundary at
  about 6|Δμ|, and their trait score saturates far above the POS level.
- But neither practical "full-flow" estimator fixes this in 256-d.
  - Guidance has no repulsion term. It pulls points toward high-density regions of p̂+ and over-concentrates them:
    MMD rises while the points stay in support. This mode-seeking behaviour is a known property of score guidance.
  - The particle flow has the repulsion term, but with a 256-d KDE it stalls at about 2.3|Δμ| of displacement.
- The flatness/boundary conclusions of ot_geometry stand: Δμ is the best simple transport and the boundary is not
  trait-specific.

## P3 — conditional (within-question) OT
Exact OT within each of the 79 held-out questions; medians over heads.

| | top-24 | random-8 |
|---|---|---|
| cos(conditional d, ∇s) | **.095** | .047 |
| cos(unconditional d, ∇s) | .129 | .078 |
| shuffle null | .067 | .028 |
| cos with w (cond / uncond) | .101 / .136 | .057 / .083 |
| cos with Δμ (cond / uncond) | .165 / .225 | .075 / .120 |
| split-half ceiling, question mean d | .32 | .32 |
| cos(Δμ, ∇s) / cos(w, ∇s) | .60 / .98 | .61 / .97 |

**Verdict: FAIL.**
- The prereg asked for ≥ +0.10 over unconditional OT on ≥ 70% of heads. Conditional OT is in fact lower than
  unconditional on 24/24 heads.
- It is above the shuffle null on 24/24 heads, so it carries some question-specific truth signal.
- Within-question displacements are dominated by answer-content differences, and question-level means are only
  moderately reproducible (ceiling .32).
- ∇s (at α = 100, ≈ w) is a direction that individual pairs barely express.

## Deviations (A2, POST HOC, fixed before the rerun)
The first P2 run was discarded for two measurement defects:
1. **Stiff KDE score.** Bandwidth CV picked the smallest grid value, and Euler integration diverged (rise ~1e10).
   - Fix: dt ≤ 0.5h²/γ, steps clipped to 0.2|Δμ|, 300 steps.
   - Each score path also gets a fixed smooth bandwidth h_smooth = median 1-NN distance.
2. **Same-question evaluation bias.** A rigid shift of held-out NEG matched held-out POS of the *same* questions below
   the floor.
   - Fix: cross-question halves.

Other notes:
- The KDE-CV bandwidth sits at the lower grid edge (c = 0.25) on all top heads. In 256-d the likelihood prefers a
  nearest-neighbour KDE.
- DSM σ follows h.
- One pw32 random-head output is missing (63/64 rows).

## Not done
- P1 GPU guidance steering (gate failed).
- Composition.
- A full-dimensional whitened metric.

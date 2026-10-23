# KL gradient flow vs frozen velocity (P2), conditional OT (P3), guidance steering (P1, gated)

Status: **v2, after adversarial Opus review.** The review found that the P2 test of the full/guidance flow is **uninformative** (implementation degenerate). v1 had read it as a falsification; that reading is withdrawn.

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

## Headline (v2)
1. **Survives: frozen velocities overshoot.** Straight Δμ, the straight probe direction w, and the frozen ∇s path all
   pass through an MMD minimum and then diverge (24/24 heads, z). They leave the data support after about 6|Δμ|, and
   the trait score saturates far above the POS level (s ≈ 9.7).
2. **P2a/P2b: INCONCLUSIVE, not falsified.** The preregistered "fail" verdicts stand as numbers. But the review showed
   that neither guidance nor particle estimator implements a working KL flow in 256-d:
   - At h_smooth, the KDE uses 620–950 of about 1,100 fit-POS rows per point, so ∇log p̂+ ≈ (μ+ − x)/h².
     Guidance then just contracts toward μ+, and the particle flow reduces to a mean-matching translation.
     That is why it tracks straight Δμ.
   - At h_cv, the KDE uses 2–12 effective neighbours. The self-weight in the particle cloud's KDE is 0.95–0.997, so
     the repulsion term ≈ 0 and the "particle flow" is pure nearest-neighbour attraction.
   - The DSM score is worse than an isotropic-Gaussian score (denoising loss 9.8 vs 3.0 at σ_smooth), because the
     noise shell is about 10× the data radius.
   - The guidance-plateau prediction was ill-posed before any data: noiseless ascent on ∇s + γ∇log p+ goes to modes,
     never to the distribution p+. That is a prereg error, not a finding.
3. **P1 gate: not passed → GPU arm not run.** This is correct by the letter of the prereg, and also on substance: there
   is no CPU evidence that guidance helps. The reason, however, is that the test was uninformative, not that the
   theory was refuted.
4. **P3 fails (survives review).** Within-question OT is no more aligned with ∇s ≈ w than unconditional OT:
   .095 vs .129 on 24/24 heads. It also loses at the level of question means (.12–.14 vs .17–.18, reviewer check).
   It is still above a shuffle null (.067).
5. **Withdrawn from v1.**
   - "NEG→POS is not a translation (stalls at 2.4× floor)". The floor was wrong: true held-out POS of half A vs half B
     is already 1.5–2.4× that floor (question clustering of ~40-question halves). Against POS_A↔POS_B, Δμ's leftover
     gap is about 0–0.6 floor units.
   - "The KL flow does not close it in 256-d". Not tested.
   - "Guidance stays in support / is mode-seeking" as a finding. It is trivial for a contraction toward μ+.
   - "Particle s ≈ 1.3, no overshoot" as KL-flow evidence. It is a mean-matcher.

**NEXT (spec for a P2 v2, CPU, to be preregistered before running).**
- Bandwidth chosen so the effective KDE neighbour count is 10–50; report the count per path.
- Leave-one-out repulsion or SVGD for the particle flow.
- Drop DSM, or validate it against isotropic-Gaussian and KDE baselines with annealed σ ≪ radius/√d.
- Floor = POS_A↔POS_B or a fit-POS sample matched on question count; ≥ 10 random A/B splits.
- Question-preserving nulls (within-question residual shuffle), with h recomputed per null.
- Rise reported as a continuous value plus a "no progress" class.
- A noisy Langevin variant instead of noiseless guidance.
- Apply the original gate to those results before any GPU work.

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

**Interpretation (v1 text, superseded by the Headline v2 caveats).**
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

## Adversarial review (Opus) — objections and handling
| # | objection | severity | handling |
|---|---|---|---|
| 1 | Both KDE bandwidths degenerate (global mean pull / NN snap; self-weight kills repulsion) | fatal for the P2 guidance/particle claims | P2 relabelled inconclusive; v2 spec |
| 2 | Floor unreachable for ~40-question clouds; POS_A vs POS_B is already 1.5–2.4× | fatal for "not a translation" | claim withdrawn |
| 3 | Nulls lose question structure; h not recomputed per null | major | stated; question-preserving nulls in the v2 spec |
| 4 | DSM worse than an isotropic Gaussian score | major | DSM results flagged invalid |
| 5 | Noiseless guidance can never plateau at the distribution | major | stated as a prereg error |
| 6 | Rise metric degenerate when the min is at step 0 | major | stated; continuous rise + "no progress" class in v2 |
| 7 | Convergence fine; dt caps and clipping do not bind | minor | noted |
| 9 | P3 holds; say "vs w" | minor | headline says ∇s ≈ w |
| 10 | Forking paths in A2 (each change helped the score paths, which still failed) | major | noted; low false-negative risk, but the fixes were degenerate |
| 11 | Gate decision | minor | correct; P1 not run |

## Not done
- P1 GPU guidance steering (gate failed).
- Composition.
- A full-dimensional whitened metric.

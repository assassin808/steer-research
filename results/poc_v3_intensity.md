# PoC v3 stage 1: is there a behaviorally valid intensity coordinate? (2026-10-19)

Pre-registration: `poc_v3_intensity_design.md` (copy of ~/steer_poc/v3/DESIGN.md) (written before any run; criteria unchanged). Code: `v3.py` (GPU), `analyze.py` (CPU).
Raw outputs (not committed): `~/steer_poc/v3/out/{evil,sycophantic,optimistic}/`. Numbers below come from `results.json`.
Jobs: smoke 6325726 (8 min), evil 6326932 (47 min), sycophantic 6329755 (57 min), optimistic 6336418 (51 min).
Total about 2.7 GPU-hours on a100_3g.20gb MIG slices.

**Setup.** Qwen3-1.7B is both generator and judge (non-thinking). Sampling uses T=1 and is capped at 160 new tokens;
about 98% of responses hit the cap. TEST has 100 questions (30 own-trait, 70 other-trait) × 5 instruction paraphrases
× 6 levels per candidate, which is 500 generations per level and 9000 per trait. The level grid was fitted on 20
disjoint DEV questions, spaced equally in DEV judge score.

The three candidates:
- **(a) STEER:** add c·v_p at every position. The DEV rule picked layer 20 for evil and layer 14 for sycophantic and optimistic.
- **(b) MIX:** a geometric mixture of the neg-prompted and pos-prompted next-token distributions, with weight α in [0, 1].
- **(c) FS:** k few-shot exemplars.

The adversarial review (an Opus agent) found no code bug. It reported interpretation errors in the first draft; those are fixed below.

## Verdict

**No candidate passes all pre-registered criteria for any trait.** The headline is largely forced by C1b and C1c,
which turned out to be **infeasible as designed** (see fact 1). The ticks below are point estimates.

| trait | cand | C0 judge≈Claude ρ≥0.6 | C1a means ↑ | C1b item ρ≥0.5 | C1c per-q share≥0.7 (own-trait q) | C2 judge steps | C2 KL steps | C2 KL(p_k‖p_0) ↑ | C3 cos>0 | C4 fluent | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| evil | steer L20 | ✗ 0.52 | ✗ | ✗ 0.02 | ✗ 0.25 (0.37) | ✗ | ✗ | ✓ | ✓ | ✓ | FAIL |
| evil | mix | ✗ | ✓ | ✗ 0.17 | ✗ 0.53 (0.70) | ✗ | ✗ | ✓ | ✓ | ✓ | FAIL |
| evil | fs | ✗ | ✗ | ✗ 0.05 | ✗ 0.43 (0.40) | ✗ | ✗ | ✗* | ✗* | ✓ | FAIL |
| sycophantic | steer L14 | ✓ 0.60 | ✓ | ✗ 0.18 | ✗ 0.63 (0.90) | ✗ | ✗ (first step 4.0× median) | ✓ | ✓ | ✓ | FAIL |
| sycophantic | mix | ✓ | ✓ | ✗ 0.20 | ✗ 0.65 (0.97) | ✓ | ✗ (first step 2.3× median) | ✓ | ✓ | ✓ | FAIL |
| sycophantic | fs | ✓ | ✗ | ✗ 0.14 | ✗ 0.52 (0.77) | ✗ | ✗ | ✗* | ✗ (−0.19) | ✓ | FAIL |
| optimistic | steer L14 | ✗ 0.42 | ✓ | ✗ 0.27 | ✓ 0.72 (0.90) | ✓ | ✓ | ✓ | ✓ | ✗ (top level NLL 1.438 > 1.416) | FAIL |
| optimistic | mix | ✗ | ✓ | ✗ 0.25 | ✗ 0.66 (0.83) | ✓ | ✗ (first step 2.05× median) | ✓ | ✓ | ✓ | FAIL |
| optimistic | fs | ✗ | ✗ | ✗ 0.11 | ✗ 0.38 (0.40) | ✗ | ✗ | ✗* | ✗* | ✓ | FAIL |

\* Few-shot failures come mostly from a **degenerate grid**, not a zigzag. On DEV, k exemplars barely moved the judge,
so the DEV rule produced duplicate levels: evil k = [0, 0, 0, 1, 1, 3], and optimistic has k = 4 twice. A zero step
then gives a NaN cosine and a non-increasing KL. Only sycophantic FS has a real negative cosine (−0.19, lower bound −0.30).

**With the infeasible C1b/C1c set aside**, three candidates come close:
- **optimistic STEER** fails only C4, by 0.02 nats at the top level, and the judge-validity check C0.
- **sycophantic MIX** fails only the KL-step part of C2.
- **optimistic MIX** fails only the KL-step part of C2, plus C0.

Several C1a and C2-judge steps are within noise when bootstrapped over questions:
- sycophantic STEER step 2: CI (−0.05, 0.09);
- sycophantic MIX steps 3 and 4: CIs include 0;
- evil MIX step 4: CI (−0.17, 0.30).

"Monotone in means" therefore holds for point estimates only.

## Main facts

1. **C1b and C1c were infeasible for any coordinate between these endpoints.** The reviewer simulated an oracle path
   that mixes the TEST samples from the neg prompt and the pos prompt linearly over the 6 levels:

   | | evil | sycophantic | optimistic |
   |---|---|---|---|
   | oracle pooled ρ | 0.15 | 0.23 | 0.24 |
   | oracle per-q share | 0.53 | 0.62 | 0.62 |
   | observed MIX ρ | 0.17 | 0.20 | 0.25 |
   | observed MIX share | 0.53 | 0.65 | 0.66 |

   Observed MIX sits at this ceiling, and optimistic STEER exceeds it (share 0.72). The cause is that the two prompted
   endpoints barely separate under this judge on this question mix, where 70% of questions belong to other traits. In
   particular, a sycophancy question with no user opinion gives the model nothing to agree with. The cause is not the
   number of samples; that does not change the expected pooled ρ. On own-trait questions, the per-question monotone
   share is 0.83–0.97 for STEER and MIX on sycophantic and optimistic. The pre-registration error was setting C1b and
   C1c without an oracle ceiling.

2. **The v2 failure modes are gone for the parametric coordinates, but partly by construction.**
   - KL(p_k‖p_0) increases with level for STEER and MIX on every trait.
   - Consecutive output-space step cosines are 0.80–0.99, all with positive bootstrap lower bounds.

   For MIX, log p_α is linear in α at each token up to the normaliser. STEER moves along one fixed vector. So C3 rules
   out the v2 zigzag but is weak evidence of validity. The MIX activation-space cosine in the figure is about 1 by
   construction, because the MIX activation is defined as an interpolation of the two contexts' activations; ignore it.

3. **α and c are already close to KL-uniform; the judge is what bends.** Speed sqrt(2KL) per unit of parameter is
   nearly constant along the path:
   - evil MIX, per unit α: 0.96, 1.10, 1.16, 1.14, 1.22;
   - sycophantic STEER, per unit c: 0.47–0.49.

   The big first KL steps come from the judge-equalised grid. Where the DEV judge curve was flat (evil α 0 → 0.59;
   sycophantic c 0 → 1.53), the grid had to take a large parameter jump. Equalising both judge and KL steps is possible
   in one case: optimistic STEER passes both. The KL step sizes for evil MIX are 0.57, 0.25, 0.094, 0.057, 0.061.

4. **The behavioral effect is real but small compared with between-question and sample noise.** Level-mean Qwen-judge scores (0–9), all TEST questions and own-trait questions:

   | | all TEST q | own-trait q |
   |---|---|---|
   | evil MIX | 0.15 → 1.85 | 0.18 → 3.10 |
   | sycophantic | about 4.85 → 5.38 | about 5.2 → 6.3 |
   | optimistic | 3.4 → 4.7–4.9 | — |

   Evil is bimodal per sample: the persona either switches on (judge 6–9) or not at all.

5. **The small judge is only partly valid.** An independent Claude labeller scored 108 blind items (36 per trait,
   levels 0/2/5 of each candidate). The first labelling round leaked stratum order through item IDs, and the key sat in
   the same folder. It was redone with shuffled IDs and the key moved away; the numbers below come from the redo.

   | | evil | sycophantic | optimistic |
   |---|---|---|---|
   | Spearman, Qwen judge vs Claude | 0.52 | 0.60 | 0.42 |
   | Pearson, Qwen judge vs Claude (post-hoc) | 0.91 | — | — |
   | Spearman, Claude round 1 vs round 2 | 0.88 | 0.94 | 0.83 |

   The rater-to-rater agreement shows the reference is stable. Claude's means at levels 0, 2 and 5 (n = 4 per cell):

   | | evil | sycophantic | optimistic |
   |---|---|---|---|
   | STEER | 0, 0.5, 0.5 | 2.25, 3.25, 6.0 | 3.5, 4.5, 5.75 |
   | MIX | 0, 2.5, 5.25 | 2.5, 1.5, 5.5 | 1.25, 3.5, 5.0 |
   | FS | 0, 0, 0 | 3.5, 4.25, 5.0 | 4.0, 4.5, 5.25 |

   The Qwen judge compresses sycophancy into about 4.6–6.0. It also cannot separate pessimistic from neutral: optimistic
   MIX level 0 scores 3.53 against 3.42 for neutral, while Claude gives 1.25 against about 3.5. **The judge, not the
   coordinate, is the weakest link for optimistic.**

6. **Fluency.**
   - Coherence drops by at most 0.9 on 0–9 (evil MIX, 6.58 → 5.67). rep4 stays at or below 0.02, and no response is empty.
   - Length confound: the share of responses that end naturally before the cap rises with level for evil MIX (8% → 34%)
     and sycophantic STEER (0% → 21%). The judge therefore sees answers of different completeness at different levels.

7. **Undisclosed-deviation check.** DESIGN asked for 105 blind items including references; the run used 108 items with
   no reference conditions. Isotonic grid fitting included parameters that fail the fluency check. Refitting on fluent
   parameters only gives the same layer choice. |v_p| varies 2–3× across paraphrases, so the same c means different
   strengths.

## Smallest change (v3.1; pre-register on FRESH questions before running)

- **Coordinate:** MIX with **equal-α spacing** (already about KL-uniform) over the range where the judge moves, and
  STEER with **norm-normalised v_p**, capped below the fluency limit.
- **Questions:** only questions that can express the trait, i.e. own-trait questions, and for sycophancy questions with a stated opinion.
- **Samples:** 20 samples per (question, level) on TEST, and more on DEV (DEV grid choices hinged on differences of about 0.1).
- **Judge:** replace the 0–9 Qwen digit judge where C0 fails. Options are a pairwise or thresholded judge, or a larger
  blind Claude set as the reference.
- **C1 criteria:** set relative to the oracle neg/pos-mixing ceiling, plus an effect-size floor (for example a
  level-5 minus level-0 difference of at least 1.5 on 0–9 by Claude labels).
- **Evil:** treat P(persona on) as the coordinate, with a thresholded judge.
- **Length:** cap generation at a natural length (or longer), or judge the first N tokens only.

## What stage 2 can run now

**Nothing is licensed by the pre-registration.** Exploratory only, clearly labelled:

- **(W)** Trait × context interaction and cluster-gain as a function of depth, using the two endpoints only (neg vs pos
  prompt; no intensity coordinate needed). This is the robust v1/v2 result, and it can be rerun with all-position
  patching and more K.
- **(W/A)** Behavior-only curves along MIX α (equal-α grid) on own-trait sycophantic and optimistic questions.

**Activation-space geometry along t** (curvature, speed) has to wait for v3.1:
- MIX defines no hidden state at intermediate α, so it would need teacher-forced activations of α-generated text under a fixed context.
- STEER is linear at the steer layer by construction.

## Figures
- `poc_v3_intensity_judge_vs_level.png`: Qwen-judge score against level. Solid lines are all 100 TEST questions with 95% CI; dotted lines are own-trait questions.
- `poc_v3_intensity_steps_kl.png`: judge steps and sqrt(2·KL) steps divided by the median (grey band = C2); KL(p_k‖p_0) against level.
- `poc_v3_intensity_step_cosines.png`: consecutive step cosines in output space (centred probe log-prob differences; bar = bootstrap 2.5% bound) and at the L21 residual. The MIX activation cosine is about 1 by construction.

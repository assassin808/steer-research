# steer_math — rules for any agent working here

Steering-geometry research (papers W flat?, A compose, C boundary, B few answers→person) with Zheng Wang (IDEEA author).
Math notes here; dev log `DEVLOG.md` (append every step); results in `results/`; raw outputs `~/scratch/ideea_gemma/` (not in git).
Reply to the user in Chinese only.

## Hard rules (each one comes from a mistake we already made)

1. **Model choice.** Experiment design, implementation, analysis and review run on Opus. Sonnet is allowed only for pure
   job-watching / file copying. (The Qwen ngref track-1 run and the first unified Gemma run were Sonnet-run and had to be redone.)
2. **Read the baseline's code before claiming a relation to it.** IDEEA (`~/ext/IDEEA/methods/ideea.py`, read-only):
   KMeans separately on POS and on NEG head activations — **no per-question pairing**. `min_perp`/`nearest_cluster` match
   neg↔pos clusters by a QAP that maximises *cosine coherence of the matched directions* (not a transport cost).
   `nearest_pos_neg` picks nearest pos and nearest neg centroid independently. Any formulation that "generalises IDEEA"
   must use unpaired data and must reduce to it in a stated limit — check the limit in code, not on paper.
3. **Reproduce the paper's numbers first** (done: Gemma-2-2B base TI .276 vs paper .279, IDEEA .572 vs .572). Never compare
   against a baseline you have not reproduced in the same harness.
4. **Compare at matched realised per-token KL**, not matched α or vector norm.
5. **The official TruthfulQA judges over-reward steered text** (eucref official .848 vs strict .413). Headline = strict blind
   labels on FULL answers (an earlier strict pass truncated at the first blank line and penalised IDEEA/ITI: .223→.300).
   Report refusal / "I have no comment" rate and double-negation rate for every arm.
6. **Don't count a method's own training objective as a win.** ngref optimises reference log-likelihood → its MC1 gain is not evidence.
7. **One factor at a time.** ngref vs IDEEA differed in 5 factors (site, direction source, metric, clustering, normalisation)
   at once, so nothing could be attributed. Every new arm states which single factor changed.
8. **Test causal stories before telling them.** "The 'I have no comment' refs cause hedging" was wrong (removing them: gradient cos .995).
9. **Preregister** predictions + pass/fail threshold in `results/*_prereg.md` before running; report failures as failures (see `ngref_fix_fair.md`).
10. **Seeds & CIs.** ≥3 seeds or bootstrap CI over questions before calling a difference; single-seed wins are "unconfirmed".
11. **Adversarial Opus reviewer before reporting anything to the user.**
12. **Pivot, don't patch.** If a preregistered core hypothesis fails, stop spawning fix-variants; reframe from what the failure showed.
13. **Compute:** small models only (Gemma-2-2B, Qwen ≤1.7B), MIG slices, ≤2 GPU jobs per track. Narval Slurm.
14. **Robustness of runs:** write partial reports early (agents get killed at session end); keep raw outputs in `~/scratch`; Lustre
    dirs use stripe_count 1. Don't edit `~/ext/IDEEA` or `ideea_harness/` (Qwen); Gemma harness is `ideea_harness_gemma/`.

## Current state of evidence (2026-10-20)
- Gemma strict TI (150 test q): base .193, IDEEA .300, ITI .170, ngref-fix .150 (FAIL vs IDEEA), eucref-fix .413 (unconfirmed),
  ng(judge-supervised) .510.
- No candidate intensity coordinate passed PoC v3. Bridge experiment (one-factor ladder) running.
- Open direction: manifold / optimal-transport view of unpaired pos/neg samples (see DEVLOG 2026-10-20).

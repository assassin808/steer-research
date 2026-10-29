# Persona-trait manifold geometry (paper W) — status

Status: **PARTIAL (pipeline built, no data yet)** — 2026-10-28. Prereg: `results/persona_geom_prereg.md` (DRAFT v1, TBD-PC fields
open until `results/power_curve.md` exists). Code: `persona_geom/`. Raw outputs: `~/scratch/persona_geom/<RUN>/`.

## What exists
- Trait data for 6 traits (`persona_geom/data/trait_data/`): evil / sycophantic / hallucinating from persona_vectors (extract 20 +
  eval 20 questions, instructions and eval prompt unchanged); manipulative / flattering / deceptive generated with persona_vectors'
  generator prompt (D2). 240 questions, disjoint.
- Graded prompts (5 levels), crossed design (every question × 25 cells), response-mean residuals at all 27 hidden-state layers in two
  reads (neutral context = primary, generation context), Qwen3-1.7B Yes/No judges for 6 traits + coherence, mpnet text embeddings,
  blind-label packer/scorer, geometry statistics + nulls, four steering arms + prompting, matched-gain analysis.
- Synthetic checks (`test_geom.py`, CPU): all pass (see prereg §4, §6.2), including identity of our `nearest_pos_neg` with IDEEA's code.

## CPU smoke tests (login node; Narval compute nodes are all in POWEROUT20261027 until 2026-11-03)
SMOKE_RESULTS

## Waiting on
1. `results/power_curve.md` → fill prereg §7 (Q_PER_TRAIT, S_PER_CELL, primary curvature statistic, θ-effect, power table,
   S_STEER, N_BLIND) → commit prereg v2.
2. GPUs (after 2026-11-03): `RUN=main Q_PER_TRAIT=.. S_PER_CELL=.. S_STEER=.. persona_geom/sbatch/submit.sh` (refuses while TBD remain).
3. Wording validation on GPU-generated text (prereg §2.2 rule) if the CPU validation is judged insufficient.

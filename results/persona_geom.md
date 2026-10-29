# Persona-trait manifold geometry (paper W) — status

Status: **PARTIAL (pipeline built, prereg filled, no data yet)** — 2026-10-29. Prereg: `results/persona_geom_prereg.md` (v1.1 +
amendment A1: sizes and statistics from `results/power_curve.md` v2; pilot gate). Code: `persona_geom/`. Raw outputs: `~/scratch/persona_geom/<RUN>/`.

## What exists
- Trait data for 6 traits (`persona_geom/data/trait_data/`): evil / sycophantic / hallucinating from persona_vectors (extract 20 +
  eval 20 questions, instructions and eval prompt unchanged); manipulative / flattering / deceptive generated with persona_vectors'
  generator prompt (D2). 240 questions, disjoint.
- Graded instructions with one uniform frame (slight / moderate / strong / extreme + neg-strong) and a placebo instruction at every intensity
  word; ladders are placebo-referenced contrasts. Crossed design (every question × 35 cells), response-mean residuals at all 27 hidden-state layers in two
  reads (neutral context = primary, generation context), Qwen3-1.7B Yes/No judges for 6 traits + coherence, mpnet text embeddings,
  blind-label packer/scorer (K0 and steering items), geometry statistics + cell-permutation line null (primary; wild bootstrap secondary), cell-effect κ, planted-arc power, six steering arms
  (pv, chord, geo, geo_null control, ideea1, ideea) + prompting, matched-gain analysis with bootstrap over questions.
- Synthetic checks (`test_geom.py`, CPU): all pass (prereg §4, §6.2). They include the instruction-frame confound (bare-anchored ladder
  offline .040 on a straight trait vs .000 placebo-referenced) and the identity of our `nearest_pos_neg` with IDEEA's code.
- Hook check (`check_equiv.py`, Gemma on CPU): generate-time vs teacher-forced log-probs agree to 1.5e-5 for the IDEEA and add hooks.

## CPU smoke tests (login node; Narval compute nodes are all in POWEROUT20261027 until 2026-11-03)
SMOKE_RESULTS

## Waiting on
1. GPUs (after 2026-11-03): `persona_geom/sbatch/submit_pilot.sh` (g0, ~1.5 GPU-h) → `pilot_check.json` design_ok →
   `RUN=main Q_PER_TRAIT=40 S_PER_CELL=2 S_STEER=4 persona_geom/sbatch/submit.sh` (g1→g2→g3→g4 + CPU c1, c2; ≈ 16 GPU-h).
2. Blind Claude labels (K0) after g1 (`blind.py pack gen_main 300`), and after g4 (steering pack, 240).

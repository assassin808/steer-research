# Persona-trait manifold geometry (paper W) — status

Status: **PARTIAL (pipeline built, prereg filled, no data yet)** — 2026-10-30. Prereg: `results/persona_geom_prereg.md` (v1.1 +
A1: sizes and statistics from `results/power_curve.md` v2, pilot gate; A2: noise-unbiased trend energies, vector heterogeneity,
Σ_w metric, from `reports/转向流的量化方法.md`). Code: `persona_geom/`. Raw outputs: `~/scratch/persona_geom/<RUN>/`.

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
- 2026-10-29/30, 35-cell design (A1), RUN=smoke: 2 questions/source (12 questions), 2 samples, 8-token answers, pseudo-judge beyond
  24 items. **Every stage ran end to end**: test_geom, check_equiv (gen vs teacher-forced 1.5e-5), gen main (6 shards, 840 rows), judge,
  embed, steer sweep, analyze_geom (5 reps), pilot_check, steer calib, steer eval, analyze_steer, blind pack. Numbers are meaningless
  (8 tokens); they only show the code paths work. After A2 the geometry stages (analyze_geom, pilot_check) were re-run on the same
  smoke data.
- Login-node caveats: 16 GB per-user memory cap (the first two attempts were OOM-killed in the sweep, which loads Gemma and the
  Qwen judge together; the smoke now loads the judge in bf16 on CPU); the CPU steering stages take ~16 h on the login node.
- Synthetic calibration: `test_geom.py` ALL PASS (A1 and A2 checks; prereg amendment log).

## Waiting on
1. GPUs (after 2026-11-03): `persona_geom/sbatch/submit_pilot.sh` (g0, ~1.5 GPU-h) → `pilot_check.json` design_ok →
   `RUN=main Q_PER_TRAIT=40 S_PER_CELL=2 S_STEER=4 persona_geom/sbatch/submit.sh` (g1→g2→g3→g4 + CPU c1, c2; ≈ 16 GPU-h).
2. Blind Claude labels (K0) after g1 (`blind.py pack gen_main 300`), and after g4 (steering pack, 240).

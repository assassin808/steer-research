# Pre-registration: CONFIRMATORY replication of "Euclidean reference gradient at IDEEA's heads beats IDEEA"

Written 2026-10-23, **before any run of this study** (no new direction fitted, nothing generated, nothing labelled).
Model Gemma-2-2B-it, IDEEA TruthfulQA protocol in `ideea_harness_gemma/` (raw outputs `~/scratch/ideea_gemma`), everything not stated
here is exactly as in `results/bridge_prereg.md` / `results/bridge.md` (IDEEA split: dev 395 = train 316 + val 79, test 395; IDEEA eval
prompt; greedy; 50 new tokens; bf16; realised per-token KL(steered||base) along the steered greedy answer).
Code (to be written): `ideea_harness_gemma/scripts/confirm_grad.py` (directions, calibration, specs), `confirm_blind_build.py`,
`confirm_merge.py`, `confirm_report.py`, `confirm_mc.py`, `cg_*.sbatch`.

## 0. Claim under test
Bridge (single seed, 150 test q, Sonnet labels that saw the references): strict TI of **C_inf** (Euclidean gradient of the label-free
reference objective r_fix = mean log p(kept correct refs) − mean log p(kept incorrect refs), at IDEEA's 24 heads, per-head normalised)
= .460 [.38,.54] vs IDEEA paper config (min_perp, nc 2, K 24, alpha 4) .310, at matched dev KL ≈ .52. H_inf (global norm) .440.
Known weaknesses: 4–15% double negations; labeller saw references (may favour reference style; C is trained on reference likelihood);
one seed; one labeller family; 150 questions.

## 1. Arms (single factor each vs the anchor is stated; all at matched realised KL)
| arm | definition | changes vs IDEEA paper cfg |
|---|---|---|
| base | unsteered | – |
| IDEEA s0 | paper cfg `ideea:4:24:min_perp:2` (dev-fitted, KMeans random_state 0), alpha 4 (its dev KL .518 *defines* the matched level) | anchor |
| IDEEA s1, s2 | same, KMeans random_state 1 / 2 (refit with IDEEA's own `fit_fixed_nc`, read-only import); alpha calibrated to the target dev KL | clustering seed |
| ITI | IDEEA's ITI class, K = 16 heads (the earlier dev-selected ITI), alpha calibrated | (baseline) |
| C_inf orig | the bridge direction (full train data) | — (reference only) |
| **C_inf b1, b2, b3** | Euclidean grad of r_fix at the 24 heads, per-head normalised, from bootstrap resample b of the training references (below) | confirmatory |
| H_inf orig, b1–b3 | same gradient, global normalisation | normalisation (C → H) |
| K_inf orig, b1–b3 | eucref-fix: same objective, residual L13, global norm (`fx:efsum:13`) | site |

**Seeds of the gradient arms.** The Euclidean gradient uses no unlabeled base samples (they only enter the Fisher metric, not used here),
so the requested resampling is over the reference data: bootstrap b ∈ {1,2,3}, `RandomState(20261023 + b)`: draw the 316-ish usable train
questions (≥1 kept correct and ≥1 kept incorrect ref) with replacement; within each drawn question, resample its kept correct refs and kept
incorrect refs with replacement (same counts); g_b = mean over drawn questions of (mean U_correct − mean U_incorrect). The same resampled rows
are used at the head site (C, H) and at L13 (K) (row order of `refs.npz` and `refU_L13.npz` is identical, checked). Reported: cos(g_b, g_orig).
Sanity (must pass before GPU): the identity resample reproduces the bridge files `C_inf.json`/`H_inf.json` (cos > 1−1e-9) and `efsum` (cos > 1−1e-9);
IDEEA refit with random_state 0 reproduces the `nc_2` paper directions file (max abs diff < 1e-5).

## 2. KL calibration (dev only, label-free; bridge procedure)
- Target dev KLs: **0.518 (matched)**, plus **0.30 (below)** and **0.80 (above)** for the trend.
- Strength = log-log interpolation of strength vs mean dev KL over 395-q dev sweeps (`bridge_select.matched`). Existing dev sweeps are reused for
  the orig directions (C/H: bridge sweeps; K: eucref-fix L13 sweep; IDEEA s0: alpha 2/4/6; ITI K16: alpha 2/4/6 + new alpha 8).
  New directions (b1–b3, IDEEA s1/s2) get a 395-q dev sweep at the strengths of their orig sweep (C: .62/1.18/2.30/3.82; H: .40/.88/1.78/3.79;
  K: .19/.28/.33/.37; IDEEA: 2/4/6). No judge is used on dev (nothing is selected; only strength).
- IDEEA s0 at the matched level = alpha 4 exactly (paper cfg). Test KL of every run is reported; a matched run whose test KL differs from
  IDEEA s0's by > 25% is flagged.

## 3. Test set and labels
- **All 395 TEST questions.** Each run once, after calibration, no changes. Existing deterministic runs are reused where the spec is identical
  (base, IDEEA s0, C/H/K orig at matched).
- **Strict blind labels on FULL parsed answers** (no clean()), Claude **Opus** subagents (the earlier labels were Sonnet), two conditions:
  - **(a) refs shown**: the bridge labeller prompt (`labeller_prompt_fix.txt`) with the references.
  - **(b) no refs**: same definitions, no reference answers in the file; the labeller judges truth from its own knowledge and the question.
  Both prompts also ask for a third binary field `double_negation` (confusing double/multiple negation whose literal reading differs from the
  apparent intent). Prompts archived as `ideea_harness_gemma/blind/labeller_prompt_confirm_{a,b}.txt` before labelling.
- Items deduplicated by (question, answer) within a condition, shuffled across runs, random ids (different per condition), key stored outside
  the labelling dir, chunks of ≈220 items; the condition-(b) files contain no reference fields. Labellers are told to read only their chunk.
- Labelled runs: **matched level, all 17 runs above, conditions (a) and (b)**. Off levels (0.30, 0.80): IDEEA s0, ITI, C orig, H orig, K orig
  (10 runs), **condition (b) only** (trend, descriptive). Test-retest: 300 random unique matched-level items re-labelled in each condition
  (different chunking) → within-condition kappa, to read the (a)-vs-(b) agreement against.
- Official 7B judges (one at a time) on every test run: secondary metric. MC1/MC2 (judge-free, `gm_mc.py` under the same hooks) for the
  matched-level runs: secondary, **never counted as a win** (C/H/K optimise reference likelihood, rule 6).

## 4. Per-arm report
truth, info, TI (strict, per condition; official); double-negation rate (regex `degen` of `gm_fix_report.py` applied to the FULL answer = primary
definition, the one behind the "4–15%"; labeller flag = secondary); TI with double-negation answers counted as false (both definitions);
"I have no comment" rate on full answers (whole answer is an abstention: `is_ref` on the full text; and "contains 'no comment'" anywhere);
mean words (full answer); 50-token-limit rate; realised test KL; MC1/MC2; per-seed numbers; (a)-vs-(b) agreement (kappa truthful/informative/TI on
unique items, run-level Spearman); fresh-question sensitivity = the 245 test questions NOT in the bridge's 150-question strict set.

## 5. Pass criterion (confirmatory; single primary contrast)
Let C̄ = per-question mean over seeds b1–b3 of the item TI of C_inf (each seed at its own matched strength), IDEEA = IDEEA s0 (paper cfg).
Δ = mean over 395 questions of (C̄ − IDEEA); paired bootstrap over questions (10 000 resamples), 95% percentile CI.
- **K1** Δ > 0 with CI excluding 0 under condition (a).
- **K2** Δ > 0 with CI excluding 0 under condition (b).
- **K3** each seed separately: Δ_b > 0 (point estimate) under both (a) and (b) (6/6).
- **K4** with regex-double-negation answers counted as false (item TI := 0 if flagged), Δ > 0 with CI excluding 0 under both (a) and (b).
**PASS = K1 ∧ K2 ∧ K3 ∧ K4.** Anything else = FAIL, reported as such, with which criteria failed. (Prediction written now: K1 passes; K2 and K4
are the risks.)
Robustness (reported, not part of PASS): C̄ vs mean of IDEEA s0–s2; Δ on the 245 fresh questions; Δ with the labeller's double-negation flag;
C at dev KL 0.30 vs IDEEA at 0.518 (rules out "C wins only because of extra KL" if positive); official-judge Δ.
Secondary contrasts (exploratory; reported with uncorrected CIs and a Holm-adjusted p over this listed set): H̄ − IDEEA, K̄ − IDEEA, C̄ − H̄,
C̄ − K̄, C̄ − ITI, IDEEA − base, C̄ − base, each in (a) and (b).

## 6. Optional second model
IDEEA's README lists llama2_7b, llama3_8b, qwen2.5_7b, mistral_7b, gemma2_2b, gemma2_9b: Gemma-2-2B is the only ≤2B model the IDEEA paper
covers (the Qwen3-1.7B harness `ideea_harness/` is our own port, not a paper model, may not be edited, and has no official-judge reproduction).
**Skipped.**

## 7. Risks written in advance
- The labeller in (b) may know TruthfulQA (memorised references) — (b) removes the reference *text*, not prior knowledge.
- Opus labels for a direction trained on reference likelihood: style bias may survive in (b) if Opus prefers hedged, reference-like phrasing.
- The regex double-negation definition is crude ("No, it does not" counts); it is used because it is the definition behind the known 4–15%;
  it penalises arms that answer with "No, ..." more.
- Power: paired SE on 395 q ≈ 0.022 for a single run; with 3 seeds averaged slightly lower; |Δ| < ~0.05 is not resolvable.
- Multiple comparisons: one confirmatory contrast (4 joint criteria); everything else is exploratory.
- Compute: ≤ 2 GPU jobs (A100 3g.20gb MIG), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, bs ≤ 8; judges one at a time.

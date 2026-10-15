# Track 1 — natural-gradient steering vs IDEEA and baselines in IDEEA's TruthfulQA harness

Date: 2026-10-15. Model: Qwen/Qwen3-1.7B (instruct, non-thinking chat template). Code: `ideea_harness/` (copy of IDEEA commit 07ced80, patches listed at the end). All numbers: IDEEA's test split (seed 0, 395 questions), greedy decoding, 50 new tokens, IDEEA's prompt (instruction + 6-shot QA in the user turn, assistant turn continued from "A: ").

## What was compared

| method | where | fitted on | strength knob |
|---|---|---|---|
| base | – | – | – |
| IDEEA (nearest_pos_neg, nc=4, k=48 heads) | attention-head inputs, all layers, last position | IDEEA head activations of reference correct/incorrect answers (train split) | alpha |
| ITI (k=48) | same | same, mass-mean | alpha |
| CAA (IDEEA's) | residual L14, last position | A/B multiple-choice contrast | norm |
| **ng** (ours, global natural gradient) | residual L14, IDEEA's CAA protocol | 16 base samples x 316 train questions, judged; v = (G + lam I)^-1 g, g = Cov(s, grad log p), G = sequence Fisher of the same intervention; lam picked on the val split | norm |
| euc (ablation) | same | same data, v = g (no metric) | norm |
| ngkl / ngx (per-input) | same | ngkl: global ng with a per-input KL budget from the local Fisher G_x (16 unlabelled samples of the test question); ngx: v(x) = (G_x + lam I)^-1 g | budget eps |
| scaa (control) | same | SAME judged samples as ng; v = within-question Cov(s, mean answer-token residual) | norm |
| ngE (control) | same | as ng but s from the embedding judge | norm |
| random (3 seeds) | same | – | norm |

Strength was swept for every method (70+ runs); methods are compared at **matched realized KL**, interpolated in log-KL between neighbouring strengths. Realized KL = mean over generated tokens of KL(p_steered || p_base) along the steered greedy answer (base floor 0.002); we also report matching on first-answer-token KL, because ng puts much more of its KL on the first token (e.g. ng:0.1 per-token 0.081 vs first-token 0.91; IDEEA alpha 1: 0.071 vs 0.082).

## Judges (the 7B TruthfulQA judges are not allowed here)

- **Qwen proxy (primary):** Qwen3-1.7B asked "is the candidate true?" given the question and the reference true/false answers; P(yes) threshold (0.9998) calibrated on the DEV split's reference answers (leave-one-out). Test-split leave-one-out accuracy at the dev threshold 0.776 (balanced 0.778, AUC 0.86; `results_proxy/judge_validation_test_at_dev_threshold.json`; at the default 0.5 threshold it is 0.711). Info = not a refusal (a short-answer regex; agreement with Claude "informative" 0.73, kappa 0.43). As a check we also report TI_qq = Qwen truth AND a Qwen informativeness judge (agreement with Claude informative 0.92). Note the judge is the same 1.7B model that is being steered (judging runs with hooks removed).
- **Embedding proxy (secondary):** all-mpnet-base-v2, nearer to a correct than to an incorrect reference. Test accuracy 0.70 (AUC 0.74).
- **Blind Claude-subagent labels (validation, not a model we run):** 1400 real generations (14 runs x the same 100 test questions) + 200 earlier items, labelled truthful / informative with the TruthfulQA paper's definitions. Qwen proxy TI vs Claude TI: item agreement 0.82, kappa 0.61, run-level Spearman 0.96; embedding proxy kappa 0.41.
- **Judge-free:** TruthfulQA multiple-choice likelihood (MC1 / MC2) under the same hooks.

Fidelity: IDEEA's own CLI with our proxy judges gives base TI_qwen 0.289 (ours 0.289) and IDEEA alpha=1 0.309 (ours 0.311).

## Main result (matched per-token KL; Δ vs base with paired bootstrap SE over the 395 questions)

Base: truth_qwen 0.608, info 0.681, **TI_qwen 0.289**, TI_emb 0.362.

**KL = 0.05 per token**

| method | truth (qwen) | truth (emb) | info | TI qwen (Δ) | TI emb (Δ) |
|---|---|---|---|---|---|
| ng | 0.636 | 0.612 | 0.861 | **0.497 (+0.208±0.021)** | 0.473 (+0.111±0.021) |
| euc | 0.565 | 0.585 | 0.875 | 0.440 (+0.152±0.022) | 0.460 (+0.098±0.022) |
| ngkl (per-input budget) | 0.633 | 0.604 | 0.881 | 0.514 (+0.225±0.023) | 0.485 (+0.123±0.022) |
| ngx (per-input direction) | 0.618 | 0.594 | 0.855 | 0.473 (+0.185±0.024) | 0.449 (+0.087±0.023) |
| ITI | 0.594 | 0.673 | 0.726 | 0.319 (+0.031±0.017) | 0.399 (+0.037±0.015) |
| IDEEA | 0.573 | 0.649 | 0.730 | 0.303 (+0.014±0.016) | 0.378 (+0.016±0.016) |
| CAA | 0.586 | 0.665 | 0.680 | 0.266 (−0.023±0.014) | 0.345 (−0.017±0.013) |
| scaa (same data as ng) | 0.576 | 0.650 | 0.709 | 0.285 (−0.003±0.014) | 0.359 (−0.003±0.013) |
| random seed 0 / 1 / 2 | 0.57 / 0.63 / 0.66 | | 0.73 / 0.68 / 0.58 | +0.016 / +0.020 / −0.056 (±0.015–0.018) | +0.03 / −0.01 / −0.07 |
| ngE (fitted on emb judge) | 0.272 | 0.499 | 0.990 | 0.263 (−0.026±0.020) | 0.489 (+0.127±0.022) |

**KL = 0.2 per token**

| method | truth (qwen) | info | TI qwen (Δ) | TI emb (Δ) |
|---|---|---|---|---|
| ng | 0.656 | 0.938 | 0.595 (+0.306±0.026) | 0.525 (+0.163±0.025) |
| euc | 0.635 | 0.977 | 0.612 (+0.323±0.026) | 0.529 (+0.167±0.026) |
| ITI | 0.571 | 0.775 | 0.347 (+0.058±0.019) | 0.414 (+0.052±0.020) |
| IDEEA | 0.506 | 0.792 | 0.298 (+0.009±0.020) | 0.394 (+0.032±0.020) |
| CAA | 0.545 | 0.674 | 0.219 (−0.070±0.017) | 0.334 (−0.028±0.017) |
| scaa | 0.569 | 0.728 | 0.297 (+0.009±0.017) | 0.356 (−0.006±0.017) |
| random (3 seeds) | 0.54–0.70 | 0.50–0.78 | −0.085 … +0.036 | −0.10 … +0.04 |
| ngE | 0.177 | 0.997 | 0.175 (−0.114±0.023) | 0.470 (+0.107±0.026) |

TI_qq (Qwen truth AND Qwen info judge) at KL 0.05 / 0.2: ng +0.182±0.021 / +0.280±0.025; euc +0.148 / +0.304; ITI +0.036 / +0.049; IDEEA +0.023 / +0.009; random +0.020…+0.039 (seeds 0,1), −0.046 / −0.073 (seed 2); scaa +0.007 / +0.014; CAA −0.018 / −0.069. Re-matching on sequence KL (per-token x length) keeps the ranking (reviewer check: ng +0.157, ITI +0.022, IDEEA −0.004 at sequence KL 1).

Paired contrasts (TI_qwen): ng − ITI +0.177±0.020 (KL 0.05), +0.248±0.027 (0.2); ng − IDEEA +0.194±0.021 / +0.297±0.024; ng − euc +0.057±0.016 (0.05) but −0.017±0.019 (0.2); ngx − ng −0.023±0.016; IDEEA − random −0.002±0.016. Matched on first-token KL 0.2: ng − ITI +0.057±0.020, ng − IDEEA +0.112±0.020 (margins shrink but keep their sign). IDEEA/ITI defaults (alpha 15) are far too strong for this model (KL 4–10, TI_qwen ≈ 0).

## Independent labels and the judge-free check

Claude-subagent labels, same 100 questions per run (base TI 0.17; SE of Δ 0.02–0.05):

| run | per-token KL | Claude truthful | Claude informative | Claude TI (Δ vs base) | Qwen-proxy TI, same items |
|---|---|---|---|---|---|
| base | 0.002 | 0.78 | 0.39 | 0.17 | 0.29 |
| ng:0.05 / 0.1 / 0.2 | 0.025 / 0.081 / 0.268 | 0.70 / 0.62 / 0.54 | 0.60 / 0.70 / 0.81 | +0.13 / +0.15 / +0.18 | 0.44 / 0.54 / 0.61 |
| ngkl:3 / ngx:3 | 0.100 / 0.137 | 0.61 / 0.57 | 0.77 / 0.77 | +0.22 / +0.18 | 0.59 / 0.53 |
| euc:0.1 | 0.264 | 0.55 | 0.92 | +0.32 | 0.62 |
| ITI 0.5 / 1 | 0.037 / 0.143 | 0.78 / 0.83 | 0.42 / 0.41 | +0.03 / +0.07 | 0.31 / 0.32 |
| IDEEA 1 / 1.5 | 0.071 / 0.178 | 0.80 / 0.81 | 0.41 / 0.38 | +0.04 / +0.03 | 0.33 / 0.30 |
| CAA 0.2 | 0.145 | 0.76 | 0.37 | −0.03 | 0.24 |
| random 0.2 | 0.150 | 0.77 | 0.45 | +0.05 | 0.30 |
| ngE 0.2 | 0.251 | 0.94 | 0.16 | −0.07 | 0.14 |

Claude labels for the low-KL comparison (same 100 questions, set 3): at per-token KL ≈ 0.025, ng:0.05 +0.13±0.04 vs euc:0.02 +0.07±0.03 (paired ng − euc +0.06±0.03); euc needs KL 0.047 (euc:0.03, +0.13±0.03) to match ng:0.05. ng:0.05 − ITI 0.5 (KL 0.037) +0.10±0.04; ITI 1.5 (KL 0.36) reaches only +0.09±0.04. These Claude comparisons are paired but only approximately KL-matched.

Proxy inflation (Qwen TI − Claude TI on the same items) is larger for ng than for base or euc: base +0.12, euc:0.1 +0.13, ITI/IDEEA +0.08–0.12, but ng:0.05 / 0.1 / 0.2 +0.14 / +0.22 / +0.26. The ng direction, fitted to the Qwen judge, partly exploits judge-specific features; its Claude-measured gain is still clearly positive but smaller than the proxy says.

MC (judge-free, Δ vs base MC1 0.304 / MC2 0.491; not KL-matched): ng and ngkl 0.000±0.01 on both; euc:0.1 MC1 +0.023±0.012 (MC2 +0.015±0.011); ITI alpha 1 / 2.5 MC1 +0.025±0.010 / +0.048±0.015 (MC2 +0.035 / +0.062); IDEEA alpha 1 / 2.5 MC1 +0.013 / +0.025 (MC2 +0.017±0.007 / +0.036±0.013); CAA and random ≤ 0. Caveat: MC scores only answer candidates, so a shift that mainly suppresses the "I have no comment" start would barely register in MC.

## Reading

Solid (all three judges agree in sign and rank; paired SEs small relative to effects):
1. At equal realized KL, the global readout-consistent gradient directions (ng, and its un-preconditioned version euc) raise generated truth×info far more than IDEEA, ITI, CAA or random directions in this harness (+0.21 vs +0.01–0.03 TI_qwen at KL 0.05; Claude labels +0.13–0.18 vs +0.03–0.09). This holds under per-token, first-token and sequence-KL matching (margins shrink under first-token matching: ng − ITI +0.057±0.020 at first-token KL 0.2).
2. IDEEA's clustering adds nothing over ITI here for generation (IDEEA within the range of three random directions at every budget; ITI slightly better).

Suggestive, not solid:
3. Supervision source vs machinery: a CAA-style covariance direction on the same judged samples (scaa) does nothing until KL ≥ 0.5. This is one weak alternative (unwhitened, answer-token average, re-tokenised text), so it suggests but does not prove that the gain comes from the readout-consistent gradient through the actual intervention.
4. Natural gradient vs plain gradient: "ng" is heavily regularised (lam = 3 x trace(G)/d chosen on val; cos(ng, euc) = 0.78). ng beats euc only at small KL (qwen +0.057±0.016 at KL 0.05; Claude +0.06±0.03 at KL 0.025; embedding judge +0.013±0.015, n.s.) and not at KL ≥ 0.2 (Claude even favours euc:0.1 at KL 0.26: +0.32 vs ng:0.2 +0.18). Consistent with first-order theory, but weak evidence.
5. Per-input variants: the local metric (ngx) does not help (−0.023±0.016); a per-input KL budget (ngkl) is neutral to slightly positive (+0.017±0.011).

Caveats:
- **What improves is informativeness, not truthfulness.** Claude-truthful drops from 0.78 to 0.54–0.70 under ng; MC is unchanged. Much of the base model's failure is prompt-specific: with IDEEA's few-shot-in-user-turn prompt and its instruction to reply "I have no comment" unless certain, Qwen3-1.7B often refuses or repeats a few-shot answer ("1992 Olympics…"); ng mainly makes it answer the asked question (first-token KL is 10–30x the per-token KL). IDEEA/ITI do move truth-likelihood (MC) in the right direction, which is what they are designed for. A prompt-ablation baseline (IDEEA's instruction without the 'reply "I have no comment" unless certain' clause, unsteered) raises info to 0.85 but leaves TI unchanged (TI_qwen 0.296 vs 0.289; TI_emb 0.395 vs 0.362; answers run to the 50-token limit): simply removing the refusal pressure does not give ng's gain.
- **Asymmetric supervision and compute.** ng/euc are fitted to judged model samples (16 samples x 316 train questions + one backward pass each; judge = the evaluation proxy, with reference answers of TRAIN questions); IDEEA/ITI/CAA only see reference-answer activations. This is closer to a one-step policy gradient than to classic activation steering. ngkl/ngx also draw 16 unlabelled samples of each TEST question at test time (no labels used). The ngE control shows the danger of fitting to a proxy judge: a direction fitted to the embedding judge games that judge (+0.13 emb, −0.11 qwen, −0.07 Claude).
- Proxy judges are small; Claude labels are one LLM annotator, 100 questions per run, not exactly KL-matched. Matched-KL SEs ignore uncertainty in the KL estimates and in the interpolation.
- One model, one layer (14) for residual methods, IDEEA's prompt; IDEEA hyperparameters (nc=4, k=48) not re-tuned for a 448-head model.

## Harness changes (documented, IDEEA untouched in ~/ext/IDEEA)

`common/config.py`: `qwen3_1.7b` added (28 layers, 16 heads x 128 = 2048 = hidden). `tasks/registry.py`: lazy imports (no `datasets`/statsmodels). `cli/run_eval.py`: PERSONA_STRENGTHS inlined. `tasks/truthfulqa.py`: `JUDGE=emb|qwen|both` proxy judges (`common/proxy_judge.py`); the official 7B path kept behind `JUDGE=official`. New scripts in `ideea_harness/scripts/`: `ng_lib.py` (intervention, score features with a prompt KV cache, Rao-Blackwellised KL), `ng_fit.py`, `t1_eval.py`, `t1_summarize.py`, `t1_contrasts.py`, `t1_mc.py`, `scaa_fit.py`, `rejudge_qwen.py`, `rejudge_info.py`, `validate_judge.py`, `claude_labels_analyze.py`, `test_nglib.py` (cached log-prob = full log-prob to 2e-5; score feature · v = finite difference within noise; generation-time hook = teacher-forced hook to 1.6e-5).

Raw data: `ideea_harness/results_t1/summary.json` (every run: truth/info/TI for both judges, KL), `results_t1/contrasts.json`, `results_t1/mc_summary.json`, `results_proxy/` (judge validation, Claude labels), `results_ng/fit_L14_{qwen,emb}.json` (copies; runs_ng/ holds the raw score features and is not committed). Combined: `results/track1.json`.

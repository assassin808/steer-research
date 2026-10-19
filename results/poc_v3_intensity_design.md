# PoC v3 stage 1 — behaviorally validated intensity coordinate (pre-registration)

Written 2026-10-19 BEFORE any v3 run. Nothing below may be changed after results are seen;
deviations are logged in RESULTS.md as deviations.

## Why
PoC v2 (job 5522381): adverb-graded prompts ("To no degree at all: " ... "To an extreme degree: " + trait
instruction) are not an intensity coordinate (KL from t=0 larger to t=2 than to t=4; first step 2.5-10x the
later steps; negative cosines between consecutive steps). Stage 2 geometry (W: curvature/speed/context
dependence along t; A: composition) needs a coordinate t that is validated on BEHAVIOR first.

## Fixed setup
- Generator and judge: Qwen/Qwen3-1.7B, bf16, chat template with enable_thinking=False.
- Sampling: temperature 1.0, top_p 1.0, top_k off (pure ancestral sampling, so the Rao-Blackwellised KL below
  is the KL of the sampled distribution), max_new_tokens 160, stop at <|im_end|>/eos. Seeds fixed.
- Traits: evil, sycophantic, optimistic (benign; neg end = pessimistic). Data: persona-vectors
  (pv/data_generation, trait_data_extract + trait_data_eval): 5 pos/neg instruction paraphrases, 40 questions per
  trait, 280 questions in total.
- Split per trait (seed 0): TEST = 100 questions = 30 own-trait + 70 other-trait questions;
  DEV = 20 disjoint questions = 10 own-trait + 10 other-trait. Steering vectors, few-shot exemplars and the
  level grids are fitted on DEV only.
- Nuisance = paraphrase p in 0..4. Each (question, level) gets 5 samples, one per paraphrase
  -> 500 generations per level on TEST.

## Candidate coordinates (6 levels each, level 0 = low end)
(a) STEER: no system prompt; add c * v_p to the residual output of layer L at ALL positions (prompt and generated).
    v_p = mean response-token residual at L under pos-prompt p minus under neg-prompt p (DEV generations,
    persona-vectors recipe). Level 0 = c = 0 (= neutral, unsteered). L in {14, 20} chosen on DEV.
(b) MIX (geometric prompt mixture): next-token log p_alpha ∝ (1-alpha) log p(.|neg prompt p) + alpha log p(.|pos prompt p),
    applied at every generated token (both contexts run in parallel). alpha=0 is the neg-prompted model,
    alpha=1 the pos-prompted model. This is the exponential-family (e-geodesic) interpolation between the two
    prompted distributions at the token level.
(c) FEWSHOT: no system prompt; k previous user/assistant turns whose answers are high-judge pos-prompt answers
    to DEV questions (paraphrase p = a different exemplar subset/order). Level 0 = k = 0 (neutral).

Grid choice (pre-registered rule, DEV only): dense DEV grids — STEER c (multiple of v_p) in {0,0.5,1,1.5,2,3,4,6} at L14 and L20; MIX alpha in {0,0.1,...,1}; FEWSHOT k in {0,1,2,3,4,6,8}.
On DEV compute the level-mean judge score J(param) and coherence. Fluent-admissible params: coherence mean
>= coh(pos-prompt reference on DEV) - 1.0 (0-9 scale) and rep4 <= 0.2. Top level = admissible param with the max
isotonic-fitted J (for MIX the top is fixed at alpha=1). Levels 1..4 = params whose isotonic DEV J is closest to
the 4 equally spaced targets between J(level 0) and J(top) (linear interpolation of the continuous params; nearest
available k for FEWSHOT, duplicates allowed and reported). STEER layer = the one with the higher fluent-admissible
top J (ties -> L14).

## Measurements on TEST (per trait x candidate x level)
- Judge score s in [0,9]: Qwen3-1.7B, persona-vectors eval prompt rewritten to "single digit 0-9"; s = expected
  digit under the judge's next-token distribution restricted to "0".."9".
- Coherence judge (same model, 0-9, persona-vectors coherence definition).
- Fluency: per-token NLL of the response under the NEUTRAL model (question only, no system prompt);
  rep4 = 1 - distinct 4-grams / 4-grams; empty-response share.
- Realized KL (Rao-Blackwellised): for samples y ~ p_k, per-token mean of KL(p_k(.|y<t) || p_{k-1}(.|y<t)) (step) and
  KL(p_k || p_0) (from origin), exact over the vocabulary at each position. Step distance d_k = sqrt(2*KL_step).
- Step directions: probe set = pooled TEST responses (one per question per level, paraphrase fixed per text).
  Each probe text is scored (summed response log-prob) under every level of the candidate; per level a vector
  l_k over probe texts, centred within each question. Output-space step Delta_k = l_k - l_{k-1};
  cos(Delta_k, Delta_{k+1}) with bootstrap over questions. (Centred log-prob differences = score-function tangent of
  the path of distributions; inner product = empirical L2 on the pooled probe measure.)
  Secondary: same with the mean response-token residual at layer 21 (activation space).
- Blind subset: 105 items stratified over trait x candidate x {level 0, 2, 5} (and references), saved without
  condition labels for an independent Claude labeller (0-9, same definitions). Judge validity = Spearman(Qwen
  judge, Claude) per trait.

## Acceptance criteria (per trait x candidate; all must hold to PASS)
- C0 judge validity: Spearman(Qwen judge, Claude labels) >= 0.6 on that trait's blind items (if Claude labels are
  missing the judge is "unvalidated" and the verdict is provisional).
- C1 monotone:
  (a) level-mean judge score strictly increasing over the 5 steps (point estimates);
  (b) pooled Spearman(level, item judge score) >= 0.5;
  (c) per-question monotone share: fraction of TEST questions with Spearman(level, question-mean score) >= 0.6
      is >= 0.70 (also reported for own-trait questions only; secondary).
- C2 no jump / even steps: for both step series — judge-mean steps and sqrt(2*KL_step) — every step lies within
  [0.5, 2] x the median step. Also KL(p_k||p_0) must be increasing in k (the exact v2 failure mode).
- C3 coherent direction: all 4 consecutive output-space step cosines > 0 with bootstrap 95% lower bound > 0.
- C4 fluency at every level: coherence mean >= coherence(pos-prompt reference on TEST) - 1.0; rep4 mean <= 0.2;
  median neutral-NLL <= max(neutral-NLL of neutral responses, of pos-prompt responses) + ln 2 (i.e. perplexity
  at most 2x the worse natural reference).
A coordinate "passes" a trait if C1-C4 pass (and C0 for the judge). If none passes we say so and propose the
smallest change. Monotone + coherent + fluent but uneven steps (C2 fail only) counts as "valid up to
reparametrisation" and is reported as such, not as a pass.

## Not in stage 1
Any geometry (curvature, interaction, cluster gain) — stage 2, only on coordinates that pass.

# Pre-registration: is C's strict-TI gain the debunk template? (template controls, out-of-benchmark factual QA, non-Claude labeller)

Written 2026-10-26, **before any run of this study** (no direction fitted, nothing generated, nothing labelled; only counts of existing
dev outputs were looked at to check the template data are large enough: 200 train questions have both a C debunk answer and a base abstention).
Follows `results/confirm_grad.md` (PASS, rescoped: gain = informativeness, concentrated on debunk-answer questions; C replaces IDEEA's
"I have no comment" with a debunk template "There is no…"). Everything not stated here is as in `results/confirm_grad_prereg.md`
(Gemma-2-2B-it, IDEEA TruthfulQA protocol, IDEEA eval prompt, greedy, 50 new tokens, bf16, realised per-token KL(steered‖base) along the
steered greedy answer, 395 test questions, paired bootstrap over questions 10 000, 95% percentile CI).
Code (to be written): `ideea_harness_gemma/scripts/template_ctrl.py` (template features, directions, forced-opener eval, TriviaQA eval),
`tc_blind_build.py`, `tc_qwen_label.py`, `tc_report.py`, `tc_*.sbatch`. Raw outputs `~/scratch/ideea_gemma/{runs_ng/template_ctrl,
results_gm/*_tc, blind_tc, blind_tc_keys}`.

Question: does C make the model more truthful, or did it learn TruthfulQA's answer style?

## Part 1 — template controls (TruthfulQA test, 395 q)
### Arms
| arm | definition | single factor changed vs C̄ |
|---|---|---|
| **F1** | base model, assistant turn forced to open with `A: There is no` (continue_final_message), then greedy; answer = opener + continuation | no steering; opener forced |
| **F2** | same with the opener `There is no scientific evidence` (the most frequent 5-gram opener of C's test answers) | opener text |
| **T_md** | mean-difference "template direction" at IDEEA's 24 heads, per-head normalised, applied with IDEEA's ITI hook (as C) | direction source = template-vs-abstention *activations* |
| **T_g** | Euclidean reference-likelihood gradient at the 24 heads (C's estimator, score features, per-head normalised) on the template data instead of the TruthfulQA references | training texts only |
| reused | base, IDEEA s0 (paper cfg, α 4), C_b1–b3 (dev-matched), from confirm_grad; generations unchanged | – |

**Template data (train questions only, model outputs only).** For each of the 316 TRAIN questions q: debunk set D_q = unique full answers
of C_b1, C_b2, C_b3 on q in the confirm_grad dev sweeps at strength 2.2954 (the dev grid point nearest above the matched strengths 2.00–2.14)
that match the template regex TN of `confirm_posthoc.py`
(`^(there (is|are|'s) no|there isn't|there aren't|there is not|this is a (bit of a )?(trick|hypothetical|subjective)|it'?s (impossible|difficult) to)`);
abstention set A_q = unique full answers on q of dev base (`development_grid/base.json`) and dev IDEEA s0 α 4 (`development_confirm/br_A_kms0_4.json`)
that are pure abstentions (`is_ref` on the full answer). Q* = questions with both (200). No val or test question, no reference answer, no label.
Estimator (both T arms): g = mean_{q∈Q*} ( mean_{D_q} f − mean_{A_q} f ), within-question so the topic cancels;
f = eval-context head inputs averaged over the answer positions (`cH` of `bridge.py feats`, float32) for **T_md**; f = head score features
U = ∇_θ log p(answer | prompt, θ)|₀ (`U` of `bridge.py feats`, C's features) for **T_g**. Encoded per-head unit (bridge `save_iti_format`).
Reported: cos(T_g, C_orig) and cos(T_md, C_orig) (global, per-head effective), cos(T_md, IDEEA/ITI mass-mean).

**KL.** T_md, T_g: 395-q dev sweep at strengths {0.5, 1, 2, 4, 8} (label-free), log-log interpolation (`bridge.c_for` rule) to dev KL
**0.518 (primary, matched)**, plus 0.30 and 0.80 (trend). F1/F2 are not KL-matched; their realised KL is reported as
KL_forced = (Σ_{opener tokens} −log p_base(token | prefix)) / (number of answer tokens incl. the opener) — the forced policy puts mass 1 on the
opener and equals the base model afterwards, so continuation tokens contribute 0.

### Labels
New strict blind Opus pass, FULL answers, prompts `labeller_prompt_confirm_{a,b}.txt` unchanged, both conditions (a) refs shown, (b) no refs.
To compare within one labelling pass, the pass contains **the 4 control runs at the matched level AND re-labels base, IDEEA s0, C_b1, C_b2, C_b3
(matched)**; dedup by (question, answer) within condition, shuffled, random ids, key outside the labelling dir, chunks ≈ 220.
T_md and T_g at dev KL 0.30 / 0.80: condition (b) only (trend). Drift vs the confirm_grad labels is reported (agreement and ΔTI on identical items).

### Contrasts and criteria (primary: new-pass labels)
Δ_X = C̄ − X (strict TI; C̄ = per-question mean over b1–b3) for X ∈ {F1, F2, T_md, T_g} at the matched level, conditions (a) and (b).
- **TEMPLATE-SUFFICIENT** if for at least one X, Δ_X's CI includes 0 or Δ_X < 0, in both (a) and (b) ⇒ "the gain is reproduced by the template".
- **GRADIENT-SPECIFIC** if for every X, Δ_X > 0 with CI excluding 0 in both (a) and (b) ⇒ "C's gain is not reproduced by the template controls".
- Otherwise **MIXED** (reported per arm/condition).
Prediction written now: F1/F2 fall well short of C overall (forcing "There is no" on specific-fact questions should hurt), T_g is the arm most
likely to reach C. Reported per arm: fraction of the C̄ − IDEEA gain recovered, (X − IDEEA)/(C̄ − IDEEA); truthful / informative split;
template-opener rate; abstention rate; regex double negations; and the question-type split of `confirm_posthoc.py` (Best Answer
negation/debunk, 140 q vs specific, 255 q). Template-sufficiency on the debunk subset only is reported as a secondary result, not a verdict.
Sensitivity: the same contrasts with C̄ taken from the confirm_grad labels (cross-pass).

## Part 2 — out-of-benchmark factual QA (TriviaQA)
- **Data**: TriviaQA `rc.nocontext` validation (mandarjoshi/trivia_qa, downloaded on the login node to `~/scratch/tqa_ctrl/data_dl`),
  400 questions drawn with `RandomState(20261026)` (`~/scratch/tqa_ctrl/data/triviaqa400.json`), gold value + aliases + normalised aliases.
- **Prompt**: the harness prompt unchanged (IDEEA instruction + few-shot + `Q: question`, assistant `A: `, greedy, 50 tokens). Note: the
  instruction asks to reply "I have no comment" unless certain, so abstention is a legitimate answer mode here too.
- **Arms** (SAME directions and strengths as the TruthfulQA matched level, no refit): base; IDEEA s0 `ideea:4:24:min_perp:2`;
  ITI `iti:5.3933:16`; C_b1 `br:C_inf_b1:2.0826`, C_b2 `br:C_inf_b2:2.0045`, C_b3 `br:C_inf_b3:2.1361`; secondary: T_md, T_g at their
  matched strengths, F1. Realised KL reported per arm.
- **Metrics**: (1) **alias accuracy (primary)**: correct if any gold alias, SQuAD-normalised (lower case, punctuation and articles removed,
  whitespace collapsed), occurs as a whole-word substring of the normalised full answer; (2) abstention rate (`is_ref` on the full answer, or the
  answer starts with "I have no comment"/"I don't know"); (3) template-opener rate (regex TN); (4) wrong-answer rate = not abstention and not
  correct; (5) **adjudicated accuracy (secondary)**: blind Opus labels (question, gold aliases, answer → correct 0/1) for every answered,
  non-alias-matching item plus 100 random alias-matched items (to estimate alias false positives); adjudicated correct = Opus-correct where
  labelled, else alias match. Precision = correct / answered.
- **Criteria** (C̄ = mean over b1–b3 per question; paired bootstrap over 400 q):
  - **HURT** if C̄ − IDEEA alias accuracy < 0 with CI excluding 0. **NON-INFERIOR** if the CI lower bound ≥ −0.05. Otherwise **UNRESOLVED**.
  - the same three-way verdict for C̄ − base (reported separately).
  - Template over-generalisation: C's template-opener rate on TriviaQA (every TriviaQA question has a specific answer, so a debunk opener is
    a misapplication) — reported with CI; ≥ 0.10 is called "the template transfers to factual questions".
  - Prediction written now: C reduces abstention, raises both correct and wrong answers vs base/IDEEA, and applies the template to ≥ 10% of questions.

## Part 3 — non-Claude labeller
- **Judge**: Qwen2.5-7B-Instruct (downloaded to the HF cache; 7B, evaluation only, loaded alone, bf16, greedy, batch ≤ 4), as a blind strict
  labeller with the SAME rubric (the confirm_grad prompts (a)/(b), reworded only to one item per call and a JSON reply).
- **Primary subset (500 items)**: 125 test questions drawn with `RandomState(20261026)` × {IDEEA s0, C_b1, C_b2, C_b3} at the matched level;
  answers as generated in confirm_grad. Compared with the existing confirm_grad Opus labels of the same items (same condition).
- **Report**: Cohen's kappa (truthful, informative, TI) vs Opus per condition, overall and separately on IDEEA and C items; C̄ − IDEEA
  under Qwen labels on the 125 q (paired bootstrap), next to the same contrast under Opus labels on the same 125 q.
- **Criteria**: validity gate: kappa_TI(Qwen, Opus) ≥ 0.40 in the condition; below it the condition is **UNINFORMATIVE**.
  **SIGN-KEPT** if C̄ − IDEEA (Qwen) > 0 in both conditions; **CONFIRMED** if additionally its CI excludes 0 in both; **FLIP** if ≤ 0 in either
  valid condition.
- Secondary: Qwen labels on all matched-level answers of base, IDEEA s0, C_b1–b3 and the Part-1 control runs (full 395 q), giving a
  non-Claude version of the Part-1 contrasts (reported, not a verdict).

## Secondary: MC1/MC2 of confirm_grad (job 6902001, finished)
Filled into `results/confirm_grad.md` from `results_gm/mc_confirm` (paired bootstrap C̄ − IDEEA); never counted as a win (rule 6).

## Risks written in advance
- F1/F2 condition the model on a fixed opener, which is a stronger intervention than steering for the first tokens; their KL will differ.
- T_md is a mass-mean direction; the bridge found mass-mean ≈ orthogonal to the gradient and weaker per unit KL, so a T_md failure does not
  clear the template story by itself: T_g (same estimator as C) is the decisive control.
- The template data are C's own outputs, so T_g/T_md may also encode content of C's debunk answers, not only the template. If T_g reaches C,
  the conclusion is "C's gain is reproducible from debunk-vs-abstention outputs without the TruthfulQA references", not "pure syntax".
- The TriviaQA prompt is TruthfulQA's instruction; with another prompt the abstention rates would differ.
- Qwen-7B may know less than Opus, so in (b) its truth labels are noisier; (a) is the more reliable Qwen condition.
- Power: paired SE ≈ 0.022 (395 q, one run); ≈ 0.04–0.05 for Part 3's 125 q.
- Compute: ≤ 2 GPU jobs, A100 3g.20gb MIG, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, bs ≤ 8 (Qwen ≤ 4).

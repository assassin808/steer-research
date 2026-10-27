# Template controls, TriviaQA check, non-Claude labeller (follow-up to confirm_grad)

Date 2026-10-26/27. Pre-registration `results/template_ctrl_prereg.md` (commit 6f038a3, before any run). Numbers: `results/template_ctrl.json`.
Code: `ideea_harness_gemma/scripts/{template_ctrl.py, tc_blind_build.py, tc_qwen_label.py, tc_report.py, tc_tti_build.py, tc_*.sbatch}`;
prompts `ideea_harness_gemma/blind/labeller_prompt_{confirm_a,confirm_b,trivia,trivia_ti}.txt`. Raw outputs `~/scratch/ideea_gemma/{runs_ng/template_ctrl,
results_gm/{development,test,triviaqa}_tc, blind_tc, blind_tc_keys}`; TriviaQA subset `~/scratch/tqa_ctrl/data/triviaqa400.json`.
Status: post-review (adversarial Opus review, section "Review"). Items marked **post hoc** were added after the review and are not pre-registered.

Question: does C (Euclidean reference-likelihood gradient at IDEEA's 24 heads) make Gemma-2-2B more truthful, or did it learn TruthfulQA's answer style?

## Verdicts against the pre-registration
| part | pre-registered verdict | one line |
|---|---|---|
| 1 template controls (TruthfulQA, 395 q) | **TEMPLATE-SUFFICIENT** | forcing the base model to open with "There is no" (no steering) matches C's strict TI; so does T_g |
| 2 TriviaQA (400 q, same directions/strengths) | **HURT** vs IDEEA and vs base; template transfers (39% openers) | C answers fewer trivia questions correctly and denies that they have answers |
| 3 Qwen2.5-7B labeller (500 items) | **UNINFORMATIVE** (kappa_TI .31/.33 < .40) | Qwen is a lenient judge; sign of C − IDEEA kept (+.41/+.44) but cannot be used |
Answer to the question: **on this evidence C learned TruthfulQA's answer style.** Its strict-TI gain is reproduced by a prompt prefix,
and outside TruthfulQA the same direction makes the model less accurate and, under the same strict rubric, much less truthful.

## Part 1 — template controls (TruthfulQA test, 395 q)
Arms (each changes one factor vs C̄ = mean over the 3 bootstrap seeds of C):
F1 base + forced opener "There is no"; F2 base + "There is no scientific evidence" (most frequent 5-gram opener of C's *test* answers — a selection on
test outputs, not labels); T_md mean-difference of head activations (debunk − abstention); T_g C's estimator (score-feature gradient) on the same
template data. Template data: 200 TRAIN questions (0 overlap with test), 493 debunk answers of C_b1–b3 (dev sweep, strength 2.2954) and 200 base/IDEEA
pure abstentions (all "I have no comment."). T_g/T_md use no TruthfulQA reference text and no labels, **but they are fitted on C's outputs**, so they
are distillations of C's behaviour, not independent of the reference gradient. Only F1/F2 are fully independent of C's direction.

Directions (job 7320636): split-half reliability (per-head effective cos) T_md .997, T_g .981. **cos(T_g, C) = .50–.57**; cos(T_md, C) .03–.05;
cos(T_md, ITI mass-mean) .17–.22; cos(T_md, T_g) .02. Dev KL scans → strengths for dev KL .30/.518/.80: T_md 2.98/3.77/5.65, T_g 0.95/1.71/2.77;
realised test KL at the matched level .525 (T_md), .517 (T_g).
KL of the forced openers: KL_forced = −log p_base(opener)/answer length (exact for a point-mass forced prefix, but a different kind of divergence from C's
KL(p_steer‖p_base) over full distributions): F1 .358 per token / 8.5 nats per answer, F2 .609 / 15.8 nats, vs C .54 per token / 16.5–19.7 nats per answer.
All of F1's divergence sits in its 3 opener tokens (≈2.8 nats each; C's first-token KL is ≈3.6).

Labels: one new strict blind Opus pass (38 subagents; (a) refs shown 3 386 items, (b) no refs 4 785 items, 0 missing), which re-labels base, IDEEA s0 and
C_b1–b3 so that every contrast is within one pass. Drift vs confirm_grad labels on 1 909/1 942 identical items: agreement .954/.946, kappa TI .90/.88,
TI .372 vs .375 (a), .372 vs .369 (b).

| run | test KL | template opener | abstain (full) / "no comment" | dbl-neg rx | strict (a) T / I / **TI** | strict (b) T / I / **TI** |
|---|---|---|---|---|---|---|
| base | 0 | .00 | .70 / .71 | .00 | .87 / .28 / **.157** | .88 / .28 / **.159** |
| IDEEA s0 | .514 | .01 | .03 / .79 | .01 | .85 / .43 / **.334** | .84 / .43 / **.332** |
| C_b1 / b2 / b3 | .536 / .549 / .539 | .54 / .54 / .63 | .01–.04 / .03–.08 | .17 / .06 / .09 | .78–.84 / .48–.60 / .511, .430, .466 → **C̄ .469** | .80–.83 / .48–.61 / .524, .420, .466 → **C̄ .470** |
| **F1** base + "There is no" | .358 (forced) | 1.00 | .02 / .14 | .04 | .77 / .56 / **.453** | .75 / .60 / **.478** |
| **F2** base + "There is no scientific evidence" | .609 (forced) | 1.00 | .00 / .19 | .04 | .67 / .54 / **.453** | .68 / .60 / **.476** |
| **T_md** | .525 | .00 | .16 / .82 | .01 | .89 / .31 / **.230** | .89 / .33 / **.243** |
| **T_g** | .517 | .34 | .00 / .00 | .02 | .84 / .59 / **.471** | .85 / .57 / **.453** |
| T_md lo / hi (b only) | .329 / .901 | .00 / .00 | .29 / .06 abstain; .80 / .85 n.c. | .01 / .01 | | .90 / .32 / .225; .86 / .34 / .243 |
| T_g lo / hi (b only) | .316 / .797 | .32 / .38 | .00 / .00 abstain; .01 / .00 n.c. | .02 / .06 | | .81 / .73 / **.570**; .85 / .38 / .311 |
| *post hoc* F0 base, "comment" tokens banned | .833 | .00 | .01 / .00 (says "I have no opinion.") | .01 | .88 / .28 / .162 | .87 / .29 / .162 |
| *post hoc* FThe base + neutral opener "The" | .284 (forced) | .00 | .05 / .25 | .03 | .75 / .63 / **.405** | .76 / .65 / **.428** |

Pre-registered contrasts Δ_X = C̄ − X (strict TI; paired bootstrap over 395 q; reviewer recomputed all of them independently):
| X | Δ_X (a) | Δ_X (b) | X − IDEEA (b) | gain recovered (a / b) [CI] | truthful: C̄ − X (a / b) |
|---|---|---|---|---|---|
| F1 | +0.016 [−0.024, +0.057] | −0.008 [−0.048, +0.032] | +0.147 [+0.091, +0.203] | 0.88 [.58, 1.21] / 1.06 [.78, 1.39] | **+0.045 [+.007, +.083] / +0.054 [+.016, +.093]** |
| F2 | +0.016 [−0.027, +0.058] | −0.006 [−0.049, +0.037] | +0.144 [+0.086, +0.203] | 0.88 / 1.04 | **+0.138 / +0.125** (CIs > 0) |
| T_md | +0.239 [+0.188, +0.289] | +0.227 [+0.177, +0.276] | −0.089 [−0.137, −0.043] | −0.77 / −0.64 | −0.074 / −0.080 |
| T_g | −0.002 [−0.046, +0.042] | +0.017 [−0.024, +0.058] | +0.122 [+0.071, +0.170] | 1.01 [.70, 1.41] / 0.88 [.60, 1.19] | −0.026 / −0.037 (n.s.) |
| *post hoc* FThe | +0.064 [+0.014, +0.116] | +0.042 [−0.007, +0.094] | +0.096 [+0.046, +0.149] | 0.53 / 0.70 | +0.062 / +0.049 (CIs > 0) |
C̄ − IDEEA in this pass: +0.135 [+0.087, +0.183] (a), +0.138 [+0.090, +0.186] (b) (replicates confirm_grad's +.141/+.154).
Cross-pass sensitivity (C̄ from the confirm_grad labels): F1 +0.019 / −0.008, F2 +0.019 / −0.006, T_g +0.002 / +0.017, T_md +0.242 / +0.227 — unchanged.

Question-type split (C̄ − X; debunk-answer questions n = 140 / specific-fact questions n = 255, regex of `confirm_posthoc.py`):
| X | debunk (a) | debunk (b) | specific (a) | specific (b) |
|---|---|---|---|---|
| IDEEA | +0.338 [+0.257, +0.419] | +0.307 [+0.226, +0.390] | +0.024 [−0.033, +0.078] | +0.046 [−0.008, +0.099] |
| F1 | +0.045 [−0.021, +0.112] | +0.021 [−0.041, +0.086] | −0.000 [−0.051, +0.050] | −0.025 [−0.076, +0.024] |
| F2 | −0.076 [−0.150, −0.000] | −0.071 [−0.141, −0.002] | +0.067 [+0.017, +0.116] | +0.030 [−0.024, +0.084] |
| T_g | +0.145 [+0.074, +0.217] | +0.129 [+0.064, +0.193] | −0.082 [−0.133, −0.030] | −0.044 [−0.095, +0.007] |

**Verdict: TEMPLATE-SUFFICIENT** (pre-registered rule: at least one control whose CI of C̄ − X includes 0 in both conditions; F1, F2 and T_g all do).
What it does and does not mean:
- **F1 is the clean result.** With no steering, no reference text and no direction, the base model forced to start with "There is no" reaches C's strict
  TI in both conditions, on both question types. The prereg prediction (F1/F2 fall well short) was wrong.
- **It reproduces C's TI, not its truthfulness.** C is significantly more truthful than F1 (+.05) and F2 (+.13); the forced openers reach the same TI by
  being more informative while asserting more false things. So C is a *better-behaved* version of the template (fewer forced false denials), but its
  strict-TI advantage over IDEEA is fully available from the template prefix alone.
- **T_g matches C's total through a different question mix**, not by reproducing the template: it is worse than C on debunk questions (−.13 to −.15) and
  better on specific questions (+.04 to +.08), with only 34% template openers (26% on specific questions). T_g is mainly "stop abstaining".
  At dev KL .30, T_g (.570, b) also beats C at the same lo level (C_orig lo .425 in confirm_grad; cross-pass +.14) — post hoc, cross-pass.
- **Template vs "any non-abstention" (post hoc, after the review).** A neutral opener "The" (FThe, KL .28) recovers 53–70% of C's gain over IDEEA
  (C̄ − FThe +.064 (a, CI > 0) / +.042 (b, n.s.)). Banning the word "comment" (F0) only rewords the abstention ("I have no opinion.") and gains nothing.
  So most of the strict-TI gap between IDEEA and C is "IDEEA abstains, the others answer"; the debunk opener adds the rest (on debunk-answer questions).
- T_md reproduces nothing (0% openers, still 82% "no comment", below IDEEA): the template is reachable from the gradient/score-feature side, not from an
  activation mean-difference (cf. bridge: mass-mean ⟂ gradient).

## Part 2 — TriviaQA (400 q; same directions and strengths as TruthfulQA matched level; harness prompt)
Generations jobs 7320655, 7486381, 7490061. Opus correctness labels for **every answered item** (2 967 unique items; 15 subagents): Opus overrides alias
matching wherever it labelled. Alias matches confirmed by Opus: 90% overall (1 178 labelled), but arm-dependent (reviewer: C_b2 .45, C_b3 .50, template +
alias .25, F1's 77 alias matches are mostly "There is no record of …" denials) — hence the adjudicated column is the one to read.
| run | KL | alias acc | **adjudicated acc** | abstain / denial-type (Opus abstain) | template opener (= denial) | committed wrong (Opus, denials excluded) | wrong incl. denials |
|---|---|---|---|---|---|---|---|
| base | 0 | .422 | **.432** | .395 | .000 | .172 | .172 |
| IDEEA s0 | .566 | .388 | **.388** | .535 | .000 | .110 | .128 |
| ITI | .554 | .380 | **.390** | .500 | .000 | .135 | .160 |
| C_b1 | .658 | .290 | **.270** | .670 | .378 | .060 | .665 |
| C_b2 | .724 | .240 | **.200** | .760 | .285 | .040 | .723 |
| C_b3 | .665 | .253 | **.190** | .775 | .505 | .035 | .685 |
| F1 base + "There is no" | .576 (forced) | .193 | **.147** | .818 | 1.000 | .035 | .797 |
| T_md | .623 | .343 | **.338** | .590 | .000 | .090 | .095 |
| T_g | .593 | .510 | **.502** | .217 | .055 | .280 | .497 |
| *post hoc* F0 ("comment" banned) | .479 | .422 | .432 | .395 | .000 | .172 | .568 |
| *post hoc* FThe ("The") | .252 (forced) | .532 | **.532** | .117 | .000 | .350 | .450 |
("wrong incl. denials" = neither correct nor a regex abstention; Opus's "abstain" label includes "claims the question has no answer", so the committed-wrong
column cannot count denials as wrong by construction.)
- **Pre-registered (alias acc):** C̄ − IDEEA **−0.127 [−0.163, −0.091]**, C̄ − base −0.162 [−0.202, −0.122] → **HURT**. Template-opener rate of C̄
  .389 [.350, .429] → the template transfers to factual questions.
- Adjudicated: C̄ − IDEEA **−0.168 [−0.206, −0.131]**, C̄ − base −0.212; C̄ − ITI −0.170. Excluding aliases that appear in the question text
  (25 questions): C̄ − IDEEA −0.136.
- The forced opener F1 reproduces the damage (adjudicated .147), while T_g (which matches C on TruthfulQA) does **not** deny trivia questions (5.5%
  openers) and answers more of them correctly than base (+.12 vs IDEEA) — at the cost of more committed wrong answers (.28). C's own direction is the
  one that carries the denial template to factual questions.
- Caveat: realised KL on TriviaQA is higher for C (.66–.72) than for IDEEA (.57) at the same strengths. Per seed the damage does not track KL, and T_g at
  KL .59 is above IDEEA, so KL alone does not explain it.

*Post hoc:* the strict TruthfulQA rubric (confirm (a) prompt with gold answer + aliases as references; the prompt also tells the labeller that every
question has a known answer, a cue the TruthfulQA pass does not give) on 150 random TriviaQA q × base/IDEEA/ITI/C_b1–b3 (771 items):
truthful / informative / TI — base .800/.567/.380, IDEEA .787/.400/.313, ITI .827/.413/.327, C_b1 .433/.233/.200, C_b2 .527/.207/.167, C_b3 .373/.193/.107.
C̄ − IDEEA: truthful **−0.342 [−0.429, −0.256]**, TI −0.156 [−0.216, −0.100]. Under the rubric on which C won on TruthfulQA, C's trivia denials
("There is no widely recognized figure named Osbert Lancaster…") are false statements, so C is much *less* truthful than IDEEA outside the benchmark.

## Part 3 — Qwen2.5-7B-Instruct as blind strict labeller (jobs 7322829, 7488282)
| cond | kappa T / I / TI vs Opus | kappa TI on IDEEA / C items | Qwen TI IDEEA / C̄ | Opus TI (same items) IDEEA / C̄ | C̄ − IDEEA, Qwen | C̄ − IDEEA, Opus (same 125 q) |
|---|---|---|---|---|---|---|
| a (refs) | .19 / .28 / **.31** | .43 / .22 | .400 / .835 | .296 / .475 | +0.435 [+0.347, +0.525] | +0.179 [+0.096, +0.264] |
| b (no refs) | .07 / .36 / **.33** | .31 / .30 | .344 / .757 | .312 / .469 | +0.413 [+0.315, +0.509] | +0.157 [+0.072, +0.245] |
**Verdict: UNINFORMATIVE** (gate kappa_TI ≥ .40 failed in both conditions). Descriptively: Qwen marks 90–99% of answers truthful and counts debunk
answers as informative, like the lenient official judges (official C̄ − IDEEA +.34); it does not test a Claude style preference.
Secondary (all 395 q, 9 runs, Qwen labels; descriptive only, same gate failure): TI (b) base .225, IDEEA .334, C̄ .763, F1 .797, F2 .630, T_g .696, T_md .203;
C̄ − F1 −0.035 [−0.077, +0.008] (b) — Qwen also puts the forced-opener control level with C.
Note: Qwen2.5-7B is above the repository's ≤ 1.7B model rule (CLAUDE.md rule 13); it was used for evaluation only, like the official llama2-7B judges.
This exception was not recorded before the run and is flagged here.

## Secondary: confirm_grad MC1/MC2 (job 6902001)
Filled into `results/confirm_grad.md`: C̄ − IDEEA MC1 +0.091 [+0.057, +0.127], MC2 +0.086 [+0.054, +0.118] (expected by construction, rule 6).

## What is solid / not
Solid:
1. On TruthfulQA, C's strict-TI advantage over IDEEA (+.14) is reproduced by the base model with the fixed prefix "There is no" (Δ +.016 / −.008, CIs ±.04,
   gain recovered 88–106%), in both labelling conditions and on both question types. A neutral opener ("The") recovers 53–70%.
2. Out of benchmark, the same C directions at the same strengths lower TriviaQA accuracy by .13 (alias) / .17 (adjudicated) vs IDEEA and apply the
   debunk template to 39% of factual questions; under the strict rubric they lower truthfulness by .34 (post hoc).
3. Only the label family is unresolved: the one non-Claude judge tried is too lenient to be informative.
Not solid / caveats:
- All verdict labels are Claude Opus. The reviewer's 30-item spot check found a mild lenient tilt towards "There is no X" debunks (applies to C and F1
  alike), i.e. strict TI partly measures how much the labeller likes the debunk form.
- F1/F2/FThe/T_g are single runs compared with a 3-seed mean; per-seed C TI spans .42–.52 (e.g. C_b1 − T_g +.071 [+.023, +.119], C_b2 − F1 −.058 (b)).
  T_g has no fit bootstrap. CIs over questions understate their variance.
- The TEMPLATE-SUFFICIENT rule ("any of 4 controls fails to reject") is lenient at ±.04 power; the recovered-fraction CIs (lower bounds .58–.78 for
  F1/F2/T_g) are the better summary.
- Forced-prefix controls are a different intervention class from steering, and their KL is a different divergence (see Part 1).
- T_g/T_md are distilled from C's outputs; F2's opener was picked from C's test outputs; FThe, F0, full TriviaQA adjudication and the TriviaQA strict
  rubric are post hoc.
- The TriviaQA prompt is TruthfulQA's ("reply 'I have no comment' unless completely certain"); with another prompt abstention rates would differ.
- Unverified: F1 is generated after the assistant text "A: There is no" rather than after "A: "; the opener tokenisation was handled by a common-prefix
  rule (KL counts only the differing tail tokens).

## Review (adversarial Opus reviewer) and handling
No FATAL issue; all headline numbers and the three verdicts recomputed exactly from raw labels/keys; blind sets clean (no run names, no refs in (b),
every item labelled once, 0 test questions in the template data); no bug in the bootstrap, NEG split, mapping, recovered fraction or KL calibration.
MAJOR, all accepted:
(M1) the split claim for T_g was wrong → split table added for both conditions; T_g described as matching through a different question mix.
(M2) "F1/F2 trade truth the same way as C" was false → truthful contrasts added; the claim is now "reproduces TI, not truthfulness".
(M3) "no reference content needed" over-claimed for T_g → the claim now rests on F1; T_g/T_md described as distillations of C's outputs.
(M4) template vs "any non-abstention" confounded → post-hoc controls FThe (neutral opener) and F0 (banned "comment") run (job 7490061) and labelled.
(M5) TriviaQA adjudication incomplete and arm-biased → every answered item now Opus-labelled; alias-in-question sensitivity added; a label-id
collision across label sets (4–7 ids) found while doing this and fixed (labels are now joined per set).
(M6) "committed wrong falls" was built into the definition → "wrong incl. denials" column added; denials reported separately.
(M7) Qwen-7B vs the ≤1.7B rule → flagged above.
MINOR, handled: KL wording; seeds and verdict-leniency caveats; TriviaQA prompt cue noted; tti build script saved (`tc_tti_build.py`); stale 3.6% replaced;
Qwen secondary completed and reported; lo/hi rows now carry abstention and double-negation rates; the "a non-Claude judge favours C more" inference
removed; labeller family caveat stated.

## Next (not run)
1. Same controls on a second benchmark style (e.g. a non-adversarial QA set with a neutral instruction) to separate "anti-abstention" from "debunk".
2. A strict non-Claude labeller that passes the gate (e.g. a larger open judge on a cloud API, or human labels on 200 items).
3. If a TruthfulQA-steering claim is still wanted: train the direction against a template-matched control (debunk-form correct vs debunk-form incorrect
   answers) so the template is not part of the contrast.

## Compute
GPU (A100 3g.20gb MIG, ≤ 2 concurrent): 7320636 (0.60 h), 7320655 (0.27 h), 7322829 (0.11 h), 7486379 (0.32 h), 7486381 (0.19 h), 7488282 (0.62 h),
7490061 (0.15 h) = **2.3 GPU-h**. Labelling: Opus subagents — Part 1 38 + 8 post hoc, TriviaQA 5 + 3 + 4 + 3 + 4 (tti), ≈ 65 subagents, ≈ 13 500 labels;
1 Opus reviewer.

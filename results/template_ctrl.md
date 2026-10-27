# Template controls, TriviaQA check, non-Claude labeller (follow-up to confirm_grad)

Pre-registration: `results/template_ctrl_prereg.md` (commit 6f038a3, before any run).
Code: `ideea_harness_gemma/scripts/{template_ctrl.py, tc_blind_build.py, tc_qwen_label.py, tc_report.py, tc_*.sbatch}`;
prompts `ideea_harness_gemma/blind/labeller_prompt_{confirm_a,confirm_b,trivia}.txt`. Raw outputs `~/scratch/ideea_gemma/{runs_ng/template_ctrl,
results_gm/{development,test,triviaqa}_tc, blind_tc, blind_tc_keys}`, TriviaQA subset `~/scratch/tqa_ctrl/data/triviaqa400.json`.

Status (partial): jobs 7320636 (forced openers, template features/fit, dev KL sweep) and 7320655 (TriviaQA generations) submitted.

## Part 2 (partial, main arms) — TriviaQA 400 q, same directions and strengths as TruthfulQA, harness prompt
Generations job 7320655 (0.27 GPU-h). Opus correctness labels on 1 047 unique items (947 answered non-alias-matching + 100 alias matches; 5 subagents).
Alias matches confirmed correct by Opus: 85/100; non-matching answers judged correct: 3.6%.
| run | KL | alias acc | adjudicated acc | abstain (regex) | abstain/denial (Opus) | template opener | committed wrong (Opus) | words |
|---|---|---|---|---|---|---|---|---|
| base | 0 | .422 | .440 | .395 | .395 | .000 | .165 | 5.8 |
| IDEEA s0 | .566 | .388 | .395 | .517 | .535 | .000 | .102 | 26.3 |
| ITI | .554 | .380 | .395 | .475 | .497 | .000 | .133 | 14.3 |
| C_b1 | .658 | .290 | .302 | .065 | .650 | .378 | .052 | 22.0 |
| C_b2 | .724 | .240 | .260 | .077 | .728 | .285 | .025 | 23.7 |
| C_b3 | .665 | .253 | .268 | .125 | .725 | .505 | .020 | 20.6 |
C̄ − IDEEA alias acc **−0.127 [−0.163, −0.091]** (adjudicated −0.118 [−0.154, −0.083]); C̄ − base −0.162 [−0.202, −0.123]; template-opener rate of C̄ .389 [.350, .429].
Pre-registered verdict: **HURT** (vs IDEEA and vs base); the template transfers to factual questions (≥ .10).
Read: C answers fewer trivia questions correctly; it replaces both abstentions and right answers by denials/evasions
("There is no widely recognized figure named Osbert Lancaster…"), which Opus counts as non-answers, so committed wrong answers fall.
Caveat: realised KL on TriviaQA is higher for C (.66–.72) than for IDEEA (.57) at the same strengths.

### Part 2 post hoc (not pre-registered): the strict TruthfulQA rubric applied to TriviaQA answers
150 random TriviaQA q × base, IDEEA, ITI, C_b1–b3 (771 unique items, 4 Opus subagents, prompt `labeller_prompt_trivia_ti.txt` = confirm (a)
prompt with gold answer + aliases as reference_correct). Truthful / informative / TI:
base .800 / .567 / .380; IDEEA .787 / .400 / .313; ITI .827 / .413 / .327; C_b1 .433 / .233 / .200; C_b2 .527 / .207 / .167; C_b3 .373 / .193 / .107.
C̄ − IDEEA: truthful **−0.342 [−0.429, −0.256]**, informative −0.189 [−0.256, −0.127], TI −0.156 [−0.216, −0.100]; C̄ − base TI −0.222 [−0.289, −0.160].
Under the same strict rubric, C is far *less* truthful than IDEEA and base outside TruthfulQA: its denials ("There is no widely recognized
figure named …", "Macbeth is a fictional character…") are false statements about real things.

## Part 3 — Qwen2.5-7B-Instruct labeller (job 7322829, 0.2 GPU-h), 500 items (125 q × IDEEA s0, C_b1–b3)
| cond | kappa T / I / TI vs Opus | kappa TI on IDEEA / C items | Qwen TI IDEEA / C̄ | Opus TI (same items) IDEEA / C̄ | C̄ − IDEEA Qwen | C̄ − IDEEA Opus (same q) |
|---|---|---|---|---|---|---|
| a (refs) | .19 / .28 / **.31** | .43 / .22 | .400 / .835 | .296 / .475 | +0.435 [+0.347, +0.525] | +0.179 [+0.096, +0.264] |
| b (no refs) | .07 / .36 / **.33** | .31 / .30 | .344 / .757 | .312 / .469 | +0.413 [+0.315, +0.509] | +0.157 [+0.072, +0.245] |
Pre-registered verdict: **UNINFORMATIVE** (kappa_TI < .40 in both conditions). Descriptively the sign is kept and the gap is larger: Qwen marks
~96% of answers truthful and counts C's debunk answers as informative (I .77–.85 vs Opus ~.5), i.e. it behaves like the lenient official judges
(official C̄ − IDEEA +.34), not like a strict labeller. It cannot test a Claude style preference; if anything a non-Claude judge favours C more.

## Part 1 (partial) — template directions and forced openers
Job 7320636 (forced openers, features, fit, dev sweep; ~0.6 GPU-h). Template data: 200 train questions, 493 debunk answers (C_b1–b3 dev), 200 abstentions.
- Split-half reliability (per-head effective cos): T_md .997, T_g .981.
- **T_g (C's estimator on debunk-vs-abstention texts) has cos .50–.57 with C** (global .54–.57, per-head effective .49–.53), vs ITI mass-mean .02;
  T_md (activation mean-diff) is orthogonal to C (.03–.05) and partly aligned with the ITI mass-mean (.17–.22); T_md ⟂ T_g (.02).
  So roughly half of C's direction (by cosine) is shared with a pure "debunk template vs abstain" gradient.
- Dev KL scans (395 dev q): T_md 0.5→.005, 1→.022, 2→.118, 4→.594, 8→1.08; T_g .5→.169, 1→.315, 2→.598, 4→1.11, 8→1.66.
  Strengths for dev KL .30/.518/.80: T_md 2.98/3.77/5.65; T_g 0.95/1.71/2.77.
- Forced openers (test, not KL-matched): F1 "There is no" KL_forced .358 per token (8.5 nats per answer); F2 "There is no scientific evidence" .609 (15.8 nats).

### Part 1 results — new Opus strict pass (38 subagents; a: 3 386 items, b: 4 785 items; 0 missing)
Test gens job 7486379 (T at three KL levels, 0.2 GPU-h). Drift vs confirm_grad labels on 1 909 / 1 942 identical items: agreement .954 / .946,
kappa TI .90 / .88, TI .372 vs .375 (a), .372 vs .369 (b): the new pass is on the same scale.
| run | test KL | template opener | abstain (full) / "no comment" | dbl-neg rx | strict (a) T / I / **TI** | strict (b) T / I / **TI** |
|---|---|---|---|---|---|---|
| base | 0 | .00 | .70 / .71 | .00 | .87 / .28 / **.157** | .88 / .28 / **.159** |
| IDEEA s0 | .514 | .01 | .03 / .79 | .01 | .85 / .43 / **.334** | .84 / .43 / **.332** |
| C_b1 / b2 / b3 | .536–.549 | .54 / .54 / .63 | ≤.04 / ≤.08 | .06–.17 | C̄ **.469** | C̄ **.470** |
| **F1** base + "There is no" | .358 (forced) | 1.00 | .02 / .14 | .04 | .77 / .56 / **.453** | .75 / .60 / **.478** |
| **F2** base + "There is no scientific evidence" | .609 (forced) | 1.00 | .00 / .19 | .04 | .67 / .54 / **.453** | .68 / .60 / **.476** |
| **T_md** mid | .525 | .00 | .16 / .82 | .01 | .89 / .31 / **.230** | .89 / .33 / **.243** |
| **T_g** mid | .517 | .34 | .00 / .00 | .02 | .84 / .59 / **.471** | .85 / .57 / **.453** |
| T_md lo / hi (b) | .329 / .901 | .00 | | | | TI .225 / .243 |
| T_g lo / hi (b) | .316 / .797 | .32 / .38 | | | | TI **.570** / .311 |

Pre-registered contrasts Δ_X = C̄ − X (strict TI, 395 q, paired bootstrap):
| X | (a) | (b) | X − IDEEA (b) | gain recovered (a / b) | debunk-q Δ_X (b) | specific-q Δ_X (b) |
|---|---|---|---|---|---|---|
| F1 | +0.016 [−0.024, +0.057] | −0.008 [−0.048, +0.032] | +0.147 [+0.091, +0.203] | 88% / 106% | +0.021 [−0.041, +0.086] | −0.025 [−0.076, +0.024] |
| F2 | +0.016 [−0.027, +0.058] | −0.006 [−0.049, +0.037] | +0.144 [+0.086, +0.203] | 88% / 104% | −0.071 [−0.141, −0.002] | +0.030 [−0.024, +0.084] |
| T_md | +0.239 [+0.188, +0.289] | +0.227 [+0.177, +0.276] | −0.089 [−0.137, −0.043] | −77% / −64% | +0.450 | +0.105 |
| T_g | −0.002 [−0.046, +0.042] | +0.017 [−0.024, +0.058] | +0.122 [+0.071, +0.170] | 101% / 88% | +0.129 [+0.064, +0.193] | −0.044 [−0.095, +0.007] |
(C̄ − IDEEA in this pass: +0.135 [+0.087, +0.183] (a), +0.138 [+0.090, +0.186] (b); debunk-q +0.31–0.34, specific-q +0.02–0.05 n.s. — replicates confirm_grad.)
Cross-pass sensitivity (C̄ from confirm_grad labels): F1 +0.019 / −0.008, F2 +0.019 / −0.006, T_g +0.002 / +0.017, T_md +0.242 / +0.227 — same.

**Pre-registered verdict: TEMPLATE-SUFFICIENT.** Three of four controls reach C̄ in both conditions (CI of C̄ − X includes 0): F1, F2 and T_g.
- Forcing the base model to start with "There is no" (no steering, KL .36 — less than C's .54) gives strict TI .453/.478 = C̄ (.469/.470).
  The prediction written in the prereg (F1/F2 fall well short) was wrong. F1/F2 trade truth for informativeness the same way C does
  (truthful .67–.77 vs C̄ .80–.81; informative .54–.60 vs C̄ .53–.53).
- T_g, the same gradient estimator as C fitted on C's own *train* debunk answers vs abstentions (no TruthfulQA references, no truth labels),
  matches C at matched KL and exceeds it at dev KL .30 (.570 vs C̄ .470 (b), post-hoc comparison across levels). cos(T_g, C) ≈ .5.
- T_md (activation mean-difference) does not produce the template (0% openers, still 82% "no comment") and is below IDEEA: the template is
  reachable from the gradient/score-feature side, not from activation differences — consistent with the bridge's mass-mean ⟂ gradient result.
- Question-type split: on debunk-answer questions the template controls recover all of C's advantage (F1 Δ +.02, F2 −.07, i.e. F2 is *better*
  than C there); on specific-fact questions C ≈ F1/F2/T_g (Δ −.04..+.03, n.s.).
Read: on TruthfulQA, C's strict-TI gain over IDEEA is reproduced by a fixed opener or by a template-vs-abstention direction trained without
any reference answer; nothing in the data requires the reference gradient's TruthfulQA content.

### Part 2 secondary arms (TriviaQA; job 7486381 + 7320636; Opus labels for 526 more items, 3 subagents)
| run | KL | alias acc | adjudicated acc | abstain/denial (Opus) | template opener | committed wrong (Opus) |
|---|---|---|---|---|---|---|
| F1 base + "There is no" | .576 (forced) | .193 | .203 | .775 | 1.00 | .022 |
| T_md mid | .623 | .343 | .347 | .590 | .000 | .080 |
| T_g mid | .593 | **.510** | **.530** | .212 | .055 | .258 |
T_g − IDEEA alias acc +0.122 [+0.080, +0.165]; T_md − IDEEA −0.045 [−0.080, −0.010]; F1 − IDEEA −0.195 [−0.242, −0.147].
So the forced opener reproduces C's TriviaQA damage (denials), while T_g — which matches C on TruthfulQA — does *not* deny trivia questions
(5.5% template) and answers more of them correctly than base (and more wrongly: committed wrong .26 vs .10–.16): it is mainly an
"answer instead of abstaining" direction. C's direction is the one that carries the denial template to factual questions.

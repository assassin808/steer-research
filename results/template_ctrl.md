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

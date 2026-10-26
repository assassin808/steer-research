# Template controls, TriviaQA check, non-Claude labeller (follow-up to confirm_grad)

Pre-registration: `results/template_ctrl_prereg.md` (commit 6f038a3, before any run).
Code: `ideea_harness_gemma/scripts/{template_ctrl.py, tc_blind_build.py, tc_qwen_label.py, tc_report.py, tc_*.sbatch}`;
prompts `ideea_harness_gemma/blind/labeller_prompt_{confirm_a,confirm_b,trivia}.txt`. Raw outputs `~/scratch/ideea_gemma/{runs_ng/template_ctrl,
results_gm/{development,test,triviaqa}_tc, blind_tc, blind_tc_keys}`, TriviaQA subset `~/scratch/tqa_ctrl/data/triviaqa400.json`.

Status (partial): jobs 7320636 (forced openers, template features/fit, dev KL sweep) and 7320655 (TriviaQA generations) submitted.

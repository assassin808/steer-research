#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tc_job1_%j.log
# template-ctrl job 1: forced openers (test, triviaqa), template feats + fit, dev KL sweep of T_md / T_g
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NG_RUNS=runs_ng_confirm
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/template_ctrl.py forced --split test --opener "There is no"
$P scripts/template_ctrl.py forced --split test --opener "There is no scientific evidence"
$P scripts/template_ctrl.py forced --split triviaqa --opener "There is no"
$P scripts/template_ctrl.py feats
$P scripts/template_ctrl.py fit
$P scripts/template_ctrl.py eval --split development --tag tc --calib T_md T_g
$P scripts/template_ctrl.py cfor

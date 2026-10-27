#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=logs/tc_post_%j.log
# [post review, exploratory] neutral non-abstention controls: 'comment' tokens banned; forced neutral opener "The"
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NG_RUNS=runs_ng_confirm
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/template_ctrl.py noabst --split test
$P scripts/template_ctrl.py forced --split test --opener "The"
$P scripts/template_ctrl.py noabst --split triviaqa
$P scripts/template_ctrl.py forced --split triviaqa --opener "The"

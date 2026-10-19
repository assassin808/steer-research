#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/jobD_%j.log
# Track 1: Qwen re-judge of score-feature samples, fit NG on Qwen-judge scores, evaluation grid.
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/rejudge_qwen.py
$P scripts/ng_fit.py fit --layer 14 --judge qwen
$P scripts/ng_fit.py fit --layer 14 --judge emb
$P scripts/t1_eval.py --tag grid --specs base \
  ideea:2.5 ideea:5 ideea:10 ideea:15 iti:2.5 iti:5 iti:10 iti:15 \
  caa:0.05 caa:0.1 caa:0.2 caa:0.4 ng:0.05 ng:0.1 ng:0.2 ng:0.4 rand:0.05:0 rand:0.1:0 rand:0.2:0 rand:0.4:0 \
  euc:0.05 euc:0.1 euc:0.2 euc:0.4 ngkl:0.1 ngkl:0.3 ngkl:1 ngkl:3 ngx:0.1 ngx:0.3 ngx:1 ngx:3 caakl:0.1 caakl:0.3 caakl:1 caakl:3 \
  ideea:25 iti:25 caa:0.8 ng:0.8 rand:0.8:0 euc:0.8 ngE:0.1 ngE:0.2 ngE:0.4
$P scripts/t1_summarize.py grid

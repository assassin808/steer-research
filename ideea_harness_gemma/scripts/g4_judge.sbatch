#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/g4_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
# 1) official judges on TRAIN/VAL base samples -> judged ng fit (supervised variant)
$P scripts/gm_fit.py ngjudge --layer 13
# 2) generation for judged ng / euc
$P scripts/gm_eval.py --tag grid --specs ng:0.02 ng:0.05 ng:0.1 ng:0.2 ng:0.4 ng:0.8 euc:0.05 euc:0.2 euc:0.8
# 3) official judges (one at a time) on every unjudged generation file
$P scripts/gm_judge.py results_gm/test_grid/*.json results_gm/development_grid/*.json
# 4) judge-free MC1/MC2 under steering
$P scripts/gm_mc.py --specs base iti:2.5 iti:15 ideea:2.5 ideea:15 caa:13:5 ngref:0.05 ngref:0.2 ngref:0.8 eucref:0.2 cu:13:0.2 rand:0.2:0 ng:0.2

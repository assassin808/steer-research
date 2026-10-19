#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/g2_%j.log
# fit: unlabeled base samples (train, val) + reference gradients + ngref fit
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/gm_fit.py sample --split train --layer 13 --n 16 --max_q 2 --shard 2 && mv runs_ng/train_L13_shard000.npz runs_ng/smoke.npz
$P scripts/gm_fit.py ref --layer 13
for sp in train val; do $P scripts/gm_fit.py sample --split $sp --layer 13 --n 16; done
$P scripts/gm_fit.py fit --layer 13

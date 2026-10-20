#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/fA_%j.log
# ngref fix: per-reference score features + unlabeled base samples (seed 0) for the layer grid
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/gm_fix.py refs --layer 13 --max_q 2 --suffix _smoke
$P scripts/gm_fix.py refs --layer 13
$P scripts/gm_fix.py refs --layer 17
for sp in train val; do $P scripts/gm_fit.py sample --split $sp --layer 17 --n 16; done

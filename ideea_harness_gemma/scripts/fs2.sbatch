#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/fs2_%j.log
# unlabeled base samples, seed 2, for the ngref-fix seed refit. usage: sbatch fs2.sbatch LAYER SEED
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
export NG_RUNS=runs_ng_s$2
for sp in train val; do $P scripts/gm_fit.py sample --split $sp --layer $1 --n 16 --seed $2; done

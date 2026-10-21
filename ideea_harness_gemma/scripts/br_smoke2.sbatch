#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/br_smoke2_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 BR_SMOKE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/bridge.py fit > logs/br_smoke2_fit.txt 2>&1
$P scripts/bridge.py eval --split development --tag brsmoke2 --max_q 8 --specs br:A_inf:4 br:A_1e+12:4 br:B_inf:4 br:B_1e+12:4 br:O3_smoke:0.2 br:O2_smoke:0.2 br:I_inf:0.2 br:I_1e+12:0.2

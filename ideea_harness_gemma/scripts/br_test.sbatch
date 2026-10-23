#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/br_test_%j.log
# bridge TEST (prereg §5): matched dev-KL 0.518 strengths from bridge_selection.json, 395 test q, official judges. usage: sbatch br_test.sbatch SPEC ...
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/bridge.py eval --split test --tag bridge --specs "$@"
F=""; for s in "$@"; do F="$F results_gm/test_bridge/$(echo $s | tr ':' '_').json"; done
$P scripts/gm_judge.py $F

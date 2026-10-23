#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/cg_dev_%j.log
# confirm-grad DEV calibration (label-free KL sweeps, 395 dev q, no judges). usage: sbatch cg_dev.sbatch SPEC ...
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NG_RUNS=runs_ng_confirm
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/bridge.py eval --split development --tag confirm --specs "$@"

#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/g10_%j.log
# test-split runs of the configurations selected on the development split (best dev TI per method)
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/gm_eval.py --tag grid --specs iti:6:16 ideea:4:24:nearest_cluster:2 ideea:6:16:nearest_pos_neg:2 caa:18:1 ideea:6:16:min_perp:2
$P scripts/gm_judge.py results_gm/test_grid/*.json

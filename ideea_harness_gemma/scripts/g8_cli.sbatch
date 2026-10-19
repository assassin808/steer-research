#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/g8_%j.log
# fidelity: IDEEA's own CLI (batch_size 1, defaults alpha=15 k=48, official judges sequential) vs our pipeline; CAA layer 12 (paper Table 8)
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/gm_eval.py --tag grid --specs caa:12:1 caa:12:2 caa:12:5
$P scripts/gm_judge.py results_gm/test_grid/caa_12_*.json
$P -m cli.run_eval --task truthfulqa --model gemma2_2b --method base --seed 0 --judge_batch_size 16
$P -m cli.run_eval --task truthfulqa --model gemma2_2b --method iti --seed 0 --judge_batch_size 16
$P -m cli.run_eval --task truthfulqa --model gemma2_2b --method ideea --seed 0 --judge_batch_size 16
$P -m cli.run_eval --task truthfulqa --model gemma2_2b --method caa --caa_layer 12 --alpha 1 --seed 0 --judge_batch_size 16

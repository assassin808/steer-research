#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fxabs_%j.log
# [POST-HOC EXPLORATORY] abstention-penalised ngref-fix (L13, sum, lam 10 = pre-registered selection): calib + dev + judge
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
K="nasum10p1 nasum10p3 nasum10p10"
$P scripts/gm_fix.py calib --layer 13 --keys $K
SP=$($P scripts/gm_fix.py cfor --layer 13 --keys $K --targets 0.1 0.3 0.6 1.0)
echo "DEVSPECS $SP"
$P scripts/gm_eval.py --tag fix --split development --specs $SP
$P scripts/gm_judge.py results_gm/development_fix/fx_na*.json

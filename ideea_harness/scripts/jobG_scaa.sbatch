#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/jobG_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/scaa_fit.py
$P scripts/t1_eval.py --tag grid --specs scaa:0.02 scaa:0.05 scaa:0.1 scaa:0.2 scaa:0.4
$P scripts/rejudge_info.py
$P scripts/t1_summarize.py grid

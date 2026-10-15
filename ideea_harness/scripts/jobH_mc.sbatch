#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/jobH_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
$P scripts/t1_mc.py --specs base ng:0.05 ng:0.1 ng:0.2 euc:0.05 euc:0.1 ideea:1 ideea:2.5 iti:1 iti:2.5 caa:0.2 caa:0.4 rand:0.2:0 rand:0.4:0 ngkl:3 scaa:0.2 ideea:5 iti:5

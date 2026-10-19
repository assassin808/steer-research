#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/jobJ_%j.log
# Part A: judge-free ngref. fit -> test eval grid (tag ref) -> MC (judge-free) for ngref grid and KL-bracketing runs of other methods
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/ngref_fit.py
$P scripts/t1_eval.py --tag ref --specs ngref:0.02 ngref:0.05 ngref:0.1 ngref:0.2 ngref:0.4 ngref:0.8 \
  eucref:0.02 eucref:0.05 eucref:0.1 eucref:0.2 eucref:0.4 eucref:0.8 \
  ngrefE:0.02 ngrefE:0.05 ngrefE:0.1 ngrefE:0.2 ngrefE:0.4 ngrefE:0.8 \
  eucrefE:0.02 eucrefE:0.05 eucrefE:0.1 eucrefE:0.2 eucrefE:0.4 eucrefE:0.8
$P scripts/t1_mc.py --specs ngref:0.05 ngref:0.1 ngref:0.2 ngref:0.4 eucref:0.05 eucref:0.1 eucref:0.2 eucref:0.4 \
  euc:0.03 ideea:0.5 ideea:1.5 iti:0.5 iti:1.5 caa:0.1 rand:0.1:0 rand:0.1:1 rand:0.2:1 rand:0.4:1 rand:0.1:2 rand:0.2:2 rand:0.4:2 \
  ngrefE:0.05 ngrefE:0.1 ngrefE:0.2 ngrefE:0.4 eucrefE:0.05 eucrefE:0.1 eucrefE:0.2 eucrefE:0.4

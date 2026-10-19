#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/g5_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
SP=(base)
for a in 1 2.5 5 15; do SP+=(iti:$a ideea:$a); done
SP+=(caa:13:1 caa:13:5 caa:13:15)
for c in 0.05 0.1 0.2 0.4 0.8; do SP+=(ngref:$c eucref:$c); done
for c in 0.1 0.2 0.4; do SP+=(cu:13:$c ng:$c); done
SP+=(euc:0.2 rand:0.2:0 rand:0.2:1 rand:0.2:2 rand:0.8:0)
$P scripts/gm_mc.py --specs "${SP[@]}"

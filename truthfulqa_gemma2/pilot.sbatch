#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=pilot_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $HOME/steer_math/truthfulqa_gemma2
P=$HOME/scratch/envs/jlens/bin/python
$P -u eta_pilot.py sample --layer 14 --nq 80 --n 64 --out runs_pilot && \
$P -u eta_pilot.py fit --layer 14 --out runs_pilot && \
$P -u eta_pilot.py steer --layer 14 --nq 80 --out runs_pilot

#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=/home/yang0531/scratch/weekday/extract_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
P=$HOME/scratch/envs/jlens/bin/python
cd $HOME/steer_math/weekday
$P -u extract.py --model gemma --out $HOME/scratch/weekday
$P -u extract.py --model qwen --out $HOME/scratch/weekday

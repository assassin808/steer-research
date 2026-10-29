#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --job-name=pg_g1_gen
#SBATCH --output=/home/yang0531/scratch/persona_geom/%x_%j.log
source $HOME/steer_math/persona_geom/sbatch/_env.sh
nvidia-smi -L
retry $P -u gen.py main && retry $P -u judge.py gen_main && retry $P -u embed.py gen_main

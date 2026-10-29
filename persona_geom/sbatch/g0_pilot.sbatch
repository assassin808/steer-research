#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --job-name=pg_g0_pilot
#SBATCH --output=/home/yang0531/scratch/persona_geom/%x_%j.log
# Pilot (prereg amendment A1): 5 TRAIN + 5 TEST questions per source x 35 cells x 2 samples; design check only (kappa, SNR, H0, wording).
export RUN=pilot Q_PER_TRAIT=10 S_PER_CELL=2
source $HOME/steer_math/persona_geom/sbatch/_env.sh
nvidia-smi -L
retry $P -u gen.py main && retry $P -u judge.py gen_main

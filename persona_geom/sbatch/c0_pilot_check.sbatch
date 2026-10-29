#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --job-name=pg_c0_pilotcheck
#SBATCH --output=/home/yang0531/scratch/persona_geom/%x_%j.log
export RUN=pilot
source $HOME/steer_math/persona_geom/sbatch/_env.sh
export OMP_NUM_THREADS=8
$P -u pilot_check.py

#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --job-name=pg_c2_steer
#SBATCH --output=/home/yang0531/scratch/persona_geom/%x_%j.log
source $HOME/steer_math/persona_geom/sbatch/_env.sh
export OMP_NUM_THREADS=16
$P -u analyze_steer.py

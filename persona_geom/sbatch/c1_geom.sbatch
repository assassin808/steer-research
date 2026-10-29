#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --job-name=pg_c1_geom
#SBATCH --output=/home/yang0531/scratch/persona_geom/%x_%j.log
source $HOME/steer_math/persona_geom/sbatch/_env.sh
export OMP_NUM_THREADS=16
$P -u analyze_geom.py --layers all --split train && $P -u analyze_geom.py --layers primary --split test

#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/home/yang0531/scratch/klflow_out/logs/p3_%j.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/yang0531/scratch/steer_live/klflow
$HOME/scratch/envs/jlens/bin/python p3_condot.py

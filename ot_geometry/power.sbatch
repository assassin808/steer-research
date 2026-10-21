#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/power_%j.log
cd /home/yang0531/steer_math/ot_geometry
$HOME/scratch/envs/jlens/bin/python power.py

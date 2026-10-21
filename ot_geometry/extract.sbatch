#!/bin/bash
#SBATCH --account=def-xli135
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/extract_%j.log
cd /home/yang0531/steer_math/ot_geometry
$HOME/scratch/envs/jlens/bin/python extract.py

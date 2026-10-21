#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/yang0531/scratch/klflow_out/logs/kl_%A_%a.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/yang0531/scratch/steer_live/klflow
P=$HOME/scratch/envs/jlens/bin/python
i=$SLURM_ARRAY_TASK_ID; hi=$((i % 32)); si=$((i / 32)); SP=(z pw32); NU=(20 0)
H=$($P -c "import json;d=json.load(open('../ot_geometry/heads.json'));h=(d['top24']+d['random8'])[$hi];print(f'{h[0]}:{h[1]}')")
$P klflow.py --head $H --space ${SP[$si]} --nulls ${NU[$si]} --out /home/yang0531/scratch/klflow_out/p2

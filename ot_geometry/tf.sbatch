#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-63
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/tf_%A_%a.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/yang0531/steer_math/ot_geometry
P=$HOME/scratch/envs/jlens/bin/python
SP=(z pw32)
i=$SLURM_ARRAY_TASK_ID; hi=$((i % 32)); si=$((i / 32))
H=$($P -c "import json;d=json.load(open('heads.json'));h=(d['top24']+d['random8'])[$hi];print(f'{h[0]}:{h[1]}')")
$P trait_flow.py --heads $H --space ${SP[$si]} --reps 10 --out /home/yang0531/scratch/ot_geom/tf

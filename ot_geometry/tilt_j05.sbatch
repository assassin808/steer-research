#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --array=0-31
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/tiltj05_%A_%a.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/yang0531/steer_math/ot_geometry
P=$HOME/scratch/envs/jlens/bin/python
H=$($P -c "import json;d=json.load(open('heads.json'));h=(d['top24']+d['random8'])[$SLURM_ARRAY_TASK_ID];print(f'{h[0]}:{h[1]}')")
$P tilt_null.py --heads $H --out /home/yang0531/scratch/ot_geom/tilt_j05 --jit 0.5

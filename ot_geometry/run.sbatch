#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-63
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/run_%A_%a.log
cd /home/yang0531/steer_math/ot_geometry
P=$HOME/scratch/envs/jlens/bin/python
SP=(z pw8); REPS=(10 10)
i=$SLURM_ARRAY_TASK_ID; hi=$((i % 32)); si=$((i / 32))
H=$($P -c "import json;d=json.load(open('heads.json'));h=(d['top24']+d['random8'])[$hi];print(f'{h[0]}:{h[1]}')")
$P ot_geom.py --heads $H --space ${SP[$si]} --reps ${REPS[$si]} --out /home/yang0531/scratch/ot_geom/out --threads 4

#!/bin/bash
#SBATCH --account=def-xli135_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-95
#SBATCH --output=/home/yang0531/scratch/ot_geom/logs/tflong_%A_%a.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /home/yang0531/steer_math/ot_geometry
P=$HOME/scratch/envs/jlens/bin/python
i=$SLURM_ARRAY_TASK_ID; hi=$((i % 32)); mode=$((i / 32))
H=$($P -c "import json;d=json.load(open('heads.json'));h=(d['top24']+d['random8'])[$hi];print(f'{h[0]}:{h[1]}')")
case $mode in
 0) $P tf_long.py --heads $H --space z --out /home/yang0531/scratch/ot_geom/tflong ;;
 1) $P tf_long.py --heads $H --space pw32 --out /home/yang0531/scratch/ot_geom/tflong ;;
 2) $P tf_long.py --heads $H --space z --agrid ext --out /home/yang0531/scratch/ot_geom/tflong ;;
esac

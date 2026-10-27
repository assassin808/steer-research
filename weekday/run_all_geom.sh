#!/bin/bash
# usage: run_all_geom.sh DOM LABEL...  (single-thread BLAS: multi-thread BLAS in Pool workers stalls on the login node)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
cd $HOME/steer_math/weekday
D=$1; shift
for L in "$@"; do
  $HOME/scratch/envs/jlens/bin/python -W ignore run_geom.py --dom $D --label $L --procs 4 > $HOME/scratch/weekday/geom_${D}_${L}.log 2>&1
done

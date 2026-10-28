#!/bin/bash
# prereg tuning grid on months only (label y): k in {5,10,20} x PCA dim in {16,64,256}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
cd $HOME/steer_math/weekday
for L in y z; do for d in 16 64 256; do for k in 5 10 20; do
  [ -f $HOME/scratch/weekday/geom_gemma_month_k${k}_d${d}_${L}.json ] && continue
  $HOME/scratch/envs/jlens/bin/python -W ignore run_geom.py --dom month --label $L --k $k --d $d --reps 10 --procs 4 > $HOME/scratch/weekday/tune_k${k}_d${d}_${L}.log 2>&1
done; done; done

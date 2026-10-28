#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1
cd $HOME/steer_math/weekday
for spec in "gemma month z" "gemma weekday y" "gemma month y" "gemmaztok weekday z" "gemmaztok month z"; do
  set -- $spec
  until [ -f $HOME/scratch/weekday/$1_$2.npz ]; do sleep 30; done
  [ -f $HOME/scratch/weekday/old_$1_$2_$3.json ] && [ "$1 $2 $3" != "gemma weekday y" ] && continue
  $HOME/scratch/envs/jlens/bin/python -W ignore old_stats.py --model $1 --dom $2 --label $3 --layers 4,8,12,16,20,24 --reps 10 --procs 3 > $HOME/scratch/weekday/old_$1_$2_$3.log 2>&1
done

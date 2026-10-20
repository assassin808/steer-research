#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=logs/fdev_%j.log
# ngref fix: label-free KL calibration (40 dev q) + full DEV grid + official judges. usage: sbatch fdev.sbatch LAYER [LAYER...]
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
KN="nfsum0.1 nfsum1 nfsum10 nflen0.1 nflen1 nflen10"
for L in "$@"; do
  $P scripts/gm_fix.py fit --layer $L > logs/fit_fix_L${L}.txt
  $P scripts/gm_fix.py calib --layer $L --keys $KN efsum eflen
  SP=$($P scripts/gm_fix.py cfor --layer $L --keys $KN --targets 0.1 0.3 0.6 1.0)
  SP2=$($P scripts/gm_fix.py cfor --layer $L --keys efsum eflen --targets 0.05 0.1 0.2 0.3 0.45 0.6 0.8 1.0)
  echo "DEVSPECS L$L: $SP $SP2"
  $P scripts/gm_eval.py --tag fix --split development --specs $SP $SP2
  $P scripts/gm_judge.py results_gm/development_fix/fx_*_${L}_*.json
done

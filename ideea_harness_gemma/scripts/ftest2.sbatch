#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/ftest2_%j.log
# ngref-fix seed refits (unlabeled-sample seeds 1, 2): fit, dev-40q KL calibration to 0.518 (matched) and 0.6 (native target), TEST, judge, MC
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
for S in 1 2; do
  export NG_RUNS=runs_ng_s$S
  $P scripts/gm_fix.py fit --layer 13 > logs/fit_fix_L13_s$S.txt
  $P scripts/gm_fix.py calib --layer 13 --keys nfsum10 --tag _s$S
  SP=$($P scripts/gm_fix.py cfor --layer 13 --keys nfsum10 --targets 0.518 0.6 --tag _s$S)
  echo "SEED$S SPECS $SP"
  $P scripts/gm_eval.py --tag fix_s$S --specs $SP
  $P scripts/gm_judge.py results_gm/test_fix_s$S/*.json
  MC_SUFFIX=_s$S $P scripts/gm_mc.py --specs $SP
done

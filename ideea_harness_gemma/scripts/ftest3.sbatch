#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ftest3_%j.log
# [POST-HOC EXPLORATORY] abstention-penalised arm, dev-selected fx:nasum10p1:13 (native 0.378, matched dev-KL 0.518 c=0.3596): TEST + judge + MC
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
$P scripts/gm_eval.py --tag fix --specs fx:nasum10p1:13:0.3596 fx:nasum10p1:13:0.378 fx:nasum10p1:13:0.1834 fx:nasum10p1:13:0.2885 fx:nasum10p1:13:0.4974
$P scripts/gm_judge.py results_gm/test_fix/fx_na*.json
$P scripts/gm_mc.py --specs fx:nasum10p1:13:0.3596 fx:nasum10p1:13:0.378

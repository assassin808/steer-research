#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/g6_%j.log
# second fit seed for ngref (new unlabeled samples for G, seed 1), third random-direction seed, judging
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/gm_eval.py --tag grid --specs rand:0.05:2 rand:0.2:2 rand:0.8:2
export NG_RUNS=runs_ng_s1
$P scripts/gm_fit.py sample --split train --layer 13 --n 16 --seed 1
$P scripts/gm_fit.py sample --split val --layer 13 --n 16 --seed 1
$P scripts/gm_fit.py fit --layer 13
$P scripts/gm_eval.py --tag s1 --specs ngref:0.05 ngref:0.1 ngref:0.2 ngref:0.4 ngref:0.8 eucref:0.05 eucref:0.2
unset NG_RUNS
$P scripts/gm_judge.py results_gm/test_grid/rand_*_2.json results_gm/test_s1/*.json

#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=05:00:00
#SBATCH --output=logs/br_main_%j.log
# bridge stage 1: features, fit, sanity generations (40 dev q), label-free KL calibration (40 dev q)
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
source scripts/br_names.sh
set -ex
$P scripts/bridge.py feats
$P scripts/bridge.py fit > logs/bridge_fit.txt 2>&1
$P scripts/bridge.py eval --split development --tag brsanity --max_q 40 --specs br:A_inf:4 br:A_1e+06:4 br:B_inf:4 br:B_1e+06:4 br:Aorig_x:4 ideea:4:24:min_perp:2 br:F_hard:4 ideea:4:24:nearest_pos_neg:4 br:F_inf_inf:4 br:I_inf:0.2 br:I_1e+06:0.2 br:A_1e+12:4 br:B_1e+12:4 br:I_1e+12:0.2
$P scripts/bridge.py eval --split development --tag calibbridge --max_q 40 --calib $ALLN

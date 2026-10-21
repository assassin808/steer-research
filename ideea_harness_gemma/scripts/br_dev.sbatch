#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/br_dev_%j.log
# bridge DEV grid: strengths for dev target KLs {0.1,0.3,0.6,1.0} from the 40-q calibration, full dev split, official judges.
# usage: sbatch br_dev.sbatch NAME [NAME ...]
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
if [ -n "$SANITY" ]; then $P scripts/bridge.py eval --split development --tag brsanity --max_q 40 --specs $SANITY; fi
SP=$($P scripts/bridge.py cfor --names "$@")
echo "DEVSPECS: $SP"
$P scripts/bridge.py eval --split development --tag bridge --specs $SP
F=""; for s in $SP; do F="$F results_gm/development_bridge/$(echo $s | tr ':' '_').json"; done
$P scripts/gm_judge.py $F

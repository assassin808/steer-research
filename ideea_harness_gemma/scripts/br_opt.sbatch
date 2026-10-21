#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/br_opt_%j.log
# arm O (prereg §9): train, 40-q KL calibration, dev grid, official judges. usage: sbatch br_opt.sbatch NAME:K:OBJ:BETA ...
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
NAMES=""
for x in "$@"; do IFS=: read N K O B <<< "$x"; NAMES="$NAMES $N"
  [ -f runs_ng/bridge/dirs/$N.npz ] || $P scripts/bridge.py opt --name $N --K $K --obj $O --beta $B > logs/opt_$N.txt 2>&1; done
$P scripts/bridge.py eval --split development --tag calibbridge --max_q 40 --calib $NAMES
SP=$($P scripts/bridge.py cfor --names $NAMES)
echo "DEVSPECS: $SP"
$P scripts/bridge.py eval --split development --tag bridge --specs $SP
F=""; for s in $SP; do F="$F results_gm/development_bridge/$(echo $s | tr ':' '_').json"; done
$P scripts/gm_judge.py $F

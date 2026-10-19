#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/g3_%j.log
# generation grid on TEST (all methods), then official judges (one at a time) in bulk, then judge-free MC.
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
C="0.02 0.05 0.1 0.2 0.4 0.8"
SP=(base iti:15 ideea:15 caa:13:15)
for a in 0.5 1 2.5 5; do SP+=(iti:$a ideea:$a caa:13:$a); done
for c in $C; do SP+=(ngref:$c eucref:$c cu:13:$c); done
for c in 0.05 0.2 0.8; do SP+=(rand:$c:0 rand:$c:1); done
$P scripts/gm_eval.py --tag grid --specs "${SP[@]}"
# development-split runs of IDEEA/ITI/CAA (strength and CAA-layer selection without touching test)
DV=(base)
for a in 0.5 1 2.5 5 15; do DV+=(iti:$a ideea:$a); done
for l in 8 13 18; do for a in 1 5 15; do DV+=(caa:$l:$a); done; done
$P scripts/gm_eval.py --tag grid --split development --specs "${DV[@]}"

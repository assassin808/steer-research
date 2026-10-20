#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ftest1_%j.log
# ngref fix TEST stage (prereg section 4), selection in fix_selection.json: ngref-fix fx:nfsum10:13 (native 0.3848, matched 0.4014),
# eucref-fix fx:efsum:13 (native 0.3293, matched 0.2673); ng (supervised) and old ngref at KL~0.51; MC; official judges.
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
SP=(fx:nfsum10:13:0.4014 fx:nfsum10:13:0.3848 fx:nfsum10:13:0.1857 fx:nfsum10:13:0.2916 fx:nfsum10:13:0.5119 fx:efsum:13:0.2673 fx:efsum:13:0.3293)
for c in 0.0812 0.1132 0.1578 0.1917 0.2388 0.2804 0.3731; do SP+=(fx:efsum:13:$c); done
SP+=(ng:0.54 ngref:0.74)
$P scripts/gm_eval.py --tag fix --specs "${SP[@]}"
$P scripts/gm_judge.py results_gm/test_fix/*.json
$P scripts/gm_mc.py --specs fx:nfsum10:13:0.4014 fx:nfsum10:13:0.3848 fx:efsum:13:0.2673 fx:efsum:13:0.3293 ng:0.54 ngref:0.74

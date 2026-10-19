#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=logs/g7_%j.log
# (1) IDEEA paper hyper-parameter grid on DEV (K in {16,24}, alpha in {2,4,6}, nc 2-4, 3 variants; ITI K 8/16/24)
# (2) test: paper-reported best configs (Table 8, Gemma2 2B) and denser strength grids for our methods
# (3) official judges on everything new
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
for nc in 2 3 6; do $P -m cli.compute_directions --task truthfulqa --model gemma2_2b --method ideea --seed 0 --n_clusters $nc; done
$P scripts/gm_eval.py --tag grid --specs iti:4:24 ideea:4:24:min_perp:2 ideea:6:24:nearest_cluster:3 ngref:0.6 ngref:1.2 ngref:1.6 ngref:2.4 eucref:0.15 eucref:0.3 eucref:0.6 ng:0.3 ng:0.6 euc:0.1 euc:0.4
DV=()
for k in 8 16 24; do for a in 2 4 6; do DV+=(iti:$a:$k); done; done
for v in min_perp nearest_cluster nearest_pos_neg; do for nc in 2 3 4; do for k in 16 24; do for a in 2 4 6; do DV+=(ideea:$a:$k:$v:$nc); done; done; done; done
$P scripts/gm_eval.py --tag grid --split development --specs "${DV[@]}"
$P scripts/gm_judge.py results_gm/test_grid/*.json results_gm/development_grid/*.json

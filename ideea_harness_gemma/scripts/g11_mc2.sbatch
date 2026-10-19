#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/g11_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/gm_mc.py --specs ideea:4:24:min_perp:2 iti:4:24 iti:6:16 ideea:6:16:nearest_pos_neg:2 ideea:4:24:nearest_cluster:2 caa:12:1 ngref:0.3 eucref:0.3 ng:0.3
NG_RUNS=runs_ng_s1 MC_SUFFIX=_s1 $P scripts/gm_mc.py --specs ngref:0.2 ngref:0.4 ngref:0.8

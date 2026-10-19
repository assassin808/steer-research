#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/jobF_%j.log
# Track 1 grid 2: low-strength IDEEA/ITI (to reach matched KL), finer ng/euc, extra random seeds, IDEEA-cli fidelity check.
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/t1_eval.py --tag grid --specs ideea:0.25 ideea:0.5 ideea:1 ideea:1.5 iti:0.25 iti:0.5 iti:1 iti:1.5 \
  ng:0.02 ng:0.03 euc:0.02 euc:0.03 euc:0.1 \
  rand:0.1:1 rand:0.2:1 rand:0.4:1 rand:0.1:2 rand:0.2:2 rand:0.4:2 rand:0.8:1 \
  ngE:0.05 ngkl:0.03 ngx:0.03 caakl:10
$P scripts/rejudge_info.py
$P scripts/t1_summarize.py grid
# fidelity: IDEEA's own CLI (batch_size 1 as in their default) with the proxy judges
JUDGE=both $P -m cli.run_eval --task truthfulqa --model qwen3_1.7b --method ideea --seed 0 --alpha 1.0
JUDGE=both $P -m cli.run_eval --task truthfulqa --model qwen3_1.7b --method base --seed 0

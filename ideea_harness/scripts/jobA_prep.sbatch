#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=07:00:00
#SBATCH --output=logs/jobA_%j.log
# Track 1 prep: IDEEA activations + directions (iti, caa, ideea) for qwen3_1.7b, then our score-feature sampling.
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/ideea_harness
P=$HOME/scratch/envs/jlens/bin/python
set -x
$P scripts/ng_fit.py sample --split train --layer 14 --n 16 --max_q 2 --shard 2 && mv runs_ng/train_L14_shard000.npz runs_ng/smoke.npz
for kind in residual head; do
  [ -f results/truthfulqa/qwen3_1.7b/activations/$kind.pkl ] || $P -m cli.extract_activations --task truthfulqa --kind $kind --model qwen3_1.7b --batch_size 16 --seed 0
done
$P -m cli.compute_directions --task truthfulqa --model qwen3_1.7b --method caa --seed 0
$P -m cli.compute_directions --task truthfulqa --model qwen3_1.7b --method iti --seed 0
$P -m cli.compute_directions --task truthfulqa --model qwen3_1.7b --method ideea --seed 0
for sp in train val; do $P scripts/ng_fit.py sample --split $sp --layer 14 --n 16; done
$P scripts/ng_fit.py fit --layer 14
$P scripts/ng_fit.py sample --split test --layer 14 --n 16

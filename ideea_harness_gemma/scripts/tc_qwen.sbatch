#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/tc_qwen_%j.log
# template-ctrl Part 3: Qwen2.5-7B-Instruct strict labeller (evaluation only, loaded alone). usage: sbatch tc_qwen.sbatch PART ...
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
for part in "$@"; do for c in a b; do
  $P scripts/tc_qwen_label.py blind_tc/qwen/items_${part}_$c.json blind_tc/qwen/labels_${part}_$c.json $c
done; done

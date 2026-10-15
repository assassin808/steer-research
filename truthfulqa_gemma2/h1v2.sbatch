#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=h1v2_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/steer_math/truthfulqa_gemma2
P=$HOME/scratch/envs/jlens/bin/python
until [ -f $HOME/steer_math/ideea_harness/runs_ng/directions_L14_qwen.npz ]; do sleep 30; done
$P $HOME/steer_math/ideea_harness/scripts/test_nglib.py 2>&1 | grep -v Warn
$P -u h1v2.py run --nq 60
$P h1v2_analyze.py

#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=logs/br_smoke_%j.log
# bridge code smoke test (4 q per split; outputs in runs_ng/bridge_smoke, never used for results)
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BR_SMOKE=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -ex
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
$P scripts/bridge.py feats && echo feats-ok
$P scripts/bridge.py fit
$P scripts/bridge.py opt --K 2 --beta 10 --name O3_smoke
$P scripts/bridge.py opt --K 1 --obj correct --beta 10 --name O2_smoke
$P scripts/bridge.py eval --split development --tag brsmoke --max_q 8 --specs br:A_inf:4 br:A_1e+06:4 br:B_inf:4 br:B_1e+06:4 br:Aorig_x:4 ideea:4:24:min_perp:2 br:F_hard:4 ideea:4:24:nearest_pos_neg:4 br:F_inf_inf:4 br:F_euc0.2_1:4 br:F_mah0.05_inf:4 br:C_1:4 br:D_1:4 br:E_inf:4 br:G_1:4 br:H_0.1:4 br:I_1:0.2 br:J_inf:0.2 br:O3_smoke:0.2 br:O2_smoke:0.2

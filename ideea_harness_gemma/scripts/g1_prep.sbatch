#!/bin/bash
#SBATCH --account=def-xli135_gpu
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/g1_%j.log
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd $HOME/scratch/ideea_gemma
P=$HOME/scratch/envs/jlens/bin/python
set -x
M=gemma2_2b
$P - <<'PY'
import sys; sys.path.insert(0,'.')
from common.config import resolve_dims, MODEL_IDS
print(resolve_dims('gemma2_2b'))
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained(MODEL_IDS['gemma2_2b'])
from tasks.truthfulqa import _question_prompt
c=[{"role":"user","content":_question_prompt("Why is the sky blue?")},{"role":"assistant","content":"A: "}]
s=t.apply_chat_template(c,continue_final_message=True,tokenize=False)
print(repr(s[:60]),'...',repr(s[-120:]))
print(t.convert_tokens_to_ids('<end_of_turn>'), t.eos_token_id, t.pad_token_id)
PY
for kind in residual head; do
  [ -f results/truthfulqa/$M/activations/$kind.pkl ] || $P -m cli.extract_activations --task truthfulqa --kind $kind --model $M --batch_size 16 --seed 0
done
for m in caa iti ideea; do $P -m cli.compute_directions --task truthfulqa --model $M --method $m --seed 0; done

# sourced by every persona_geom job
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RUN=${RUN:-main}
P=$HOME/scratch/envs/jlens/bin/python
cd $HOME/steer_math/persona_geom
retry() { for i in 1 2 3; do "$@" && return 0; echo "retry $i: $*"; sleep 60; done; return 1; }

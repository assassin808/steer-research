#!/bin/bash
# End-to-end CPU smoke test of every stage. Tiny: 2 questions/trait (1 TRAIN + 1 TEST), 2 samples, 8 new tokens, pseudo-judging beyond
# 24 items per call. Tests code paths only; numbers are meaningless.
# usage: bash persona_geom/sbatch/smoke_cpu.sh   (log: ~/scratch/persona_geom/smoke/smoke.log; VALIDATE=1 also runs the wording stage)
export HF_HOME=$HOME/scratch/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export RUN=smoke SMOKE=1 MAXNEW=8 Q_PER_TRAIT=2 S_PER_CELL=2 NT=${NT:-8} OMP_NUM_THREADS=${NT:-8} NSIM_POWER=2
P=$HOME/scratch/envs/jlens/bin/python
cd $HOME/steer_math/persona_geom
set -e
st() { echo "=== $(date +%T) $*"; "$P" -u "$@" 2>&1 | grep -v -E "distutils|frozen site|Remainder of file|addpackage|Error processing line|File \"<string>\"|^$|^  Traceback \(most recent call last\):$|AttributeError: module '_distutils_hack'"; return ${PIPESTATUS[0]}; }
if [ -n "$VALIDATE" ]; then st gen.py validate; st judge.py gen_validate; fi
st test_geom.py
st check_equiv.py
st gen.py main
st judge.py gen_main
st embed.py gen_main
st steer.py sweep
st analyze_geom.py --reps 5 --nfield 2
st pilot_check.py
st steer.py calib
st steer.py eval
st analyze_steer.py
st blind.py pack gen_main 30 1
echo "=== SMOKE ALL STAGES OK $(date +%T)"

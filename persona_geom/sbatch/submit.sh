#!/bin/bash
# Submit the GPU chain (<= 1 GPU job at a time: strictly sequential via afterok). Refuses to run while the prereg still has TBD
# fields or is uncommitted (prereg rule: no GPU stage before the committed, power-curve-filled prereg).
set -e
cd $HOME/steer_math
PR=results/persona_geom_prereg.md
if grep -q "TBD" $PR; then echo "REFUSE: $PR still contains TBD fields (fill from results/power_curve.md first)"; exit 1; fi
if ! git diff --quiet HEAD -- $PR || ! git ls-files --error-unmatch $PR >/dev/null 2>&1; then echo "REFUSE: $PR not committed"; exit 1; fi
cd persona_geom/sbatch
BEGIN=${BEGIN:+--begin=$BEGIN}
EXP="--export=ALL,RUN=${RUN:-main},Q_PER_TRAIT=${Q_PER_TRAIT:?},S_PER_CELL=${S_PER_CELL:?},S_STEER=${S_STEER:?}"
j1=$(sbatch --parsable $BEGIN $EXP g1_gen.sbatch)
j2=$(sbatch --parsable $EXP --dependency=afterok:$j1 g2_sweep.sbatch)
j3=$(sbatch --parsable $EXP --dependency=afterok:$j2 g3_calib.sbatch)
c1=$(sbatch --parsable $EXP --dependency=afterok:$j2 c1_geom.sbatch)
j4=$(sbatch --parsable $EXP --dependency=afterok:$j3 g4_eval.sbatch)
c2=$(sbatch --parsable $EXP --dependency=afterok:$j4 c2_steer.sbatch)
echo "submitted gen=$j1 sweep=$j2 calib=$j3 geom(cpu)=$c1 eval=$j4 steer-analysis(cpu)=$c2"

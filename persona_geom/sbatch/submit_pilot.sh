#!/bin/bash
# Submit the pilot (one GPU job). Same prereg gate as submit.sh.
set -e
cd $HOME/steer_math
PR=results/persona_geom_prereg.md
if grep -q "TBD" $PR; then echo "REFUSE: $PR still contains TBD fields"; exit 1; fi
if ! git diff --quiet HEAD -- $PR || ! git ls-files --error-unmatch $PR >/dev/null 2>&1; then echo "REFUSE: $PR not committed"; exit 1; fi
mkdir -p $HOME/scratch/persona_geom/pilot && lfs setstripe -c 1 $HOME/scratch/persona_geom/pilot 2>/dev/null || true
cd persona_geom/sbatch; BEGIN=${BEGIN:+--begin=$BEGIN}
j0=$(sbatch --parsable $BEGIN g0_pilot.sbatch); c0=$(sbatch --parsable --dependency=afterok:$j0 c0_pilot_check.sbatch)
echo "submitted pilot gen=$j0 pilot_check(cpu)=$c0"

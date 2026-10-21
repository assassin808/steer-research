# bridge config names (prereg §3); sourced by the sbatch scripts
HEADN=""; for f in A B C D G H; do for l in 0.1 1 10 inf; do HEADN="$HEADN ${f}_$l"; done; done
HEADN="$HEADN E_inf"
FN="F_inf_inf F_inf_1"; for d in euc mah; do for t in 0.05 0.2; do for l in inf 1; do FN="$FN F_${d}${t}_$l"; done; done; done
RESN="I_0.1 I_1 I_10 I_inf J_inf"
ALLN="$HEADN $FN $RESN"

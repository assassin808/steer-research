# KL gradient flow vs frozen velocity (P2), conditional OT (P3), guidance steering (P1, gated)

**Status: PARTIAL.** P3 is done. P2 is rerunning after amendment A2 (jobs 6699270 z+nulls, 6699272 pw32).

- Prereg: `results/klflow_prereg.md`, with A1 (bandwidth grid) and A2 (post-hoc measurement fixes).
- Code: `klflow/`. Raw outputs: `~/scratch/klflow_out/`. CPU only.
- Location: `~/scratch/steer_live`, because of the /home outage.

## P3 — conditional (within-question) OT: prediction FAILS
Setup:
- Exact OT within each of the 79 held-out questions.
- s is the MLP log-ratio classifier (α = 100); w is the logistic-regression probe direction.
- Values are medians over heads of the mean cosine across held-out NEG rows, in z space.

| | top-24 | random-8 |
|---|---|---|
| cos(conditional OT d, ∇s) | **.095** | .047 |
| cos(unconditional OT d, ∇s) | .129 | .078 |
| shuffle null (NEG of q → POS of another q) | .067 | .028 |
| cos(conditional d, w) / (unconditional d, w) | .101 / .136 | .057 / .083 |
| cos(conditional d, Δμ) / (unconditional d, Δμ) | .165 / .225 | .075 / .120 |
| split-half ceiling: cos(mean d_A, mean d_B) per question | .32 | .32 |
| reference: cos(Δμ, ∇s) / cos(w, ∇s) | .60 / .98 | .61 / .97 |

**Verdict.**
- Conditional OT is *below* unconditional OT on 24/24 heads. The prereg asked for ≥ +0.10 on ≥ 70% of heads, so the
  prediction fails.
- Conditional OT is above the shuffle null on 24/24 heads, so it does carry a little question-specific truth signal.
- Within-question displacements are dominated by answer-content differences. They are only weakly reproducible within
  a question: the split-half ceiling is .32.
- The trait gradient (≈ w at α = 100) is a direction that individual pairs barely express. It is only visible
  after averaging.

## P2 — first run discarded (see prereg A2)
Two defects led to A2:
- the KDE score was stiff (CV picked the smallest bandwidth) and Euler integration diverged;
- the same-question target let straight Δμ undercut the floor (min MMD² 0.0013–0.0034 vs a floor of 0.0017–0.0034).

Logs of the discarded run are in `~/scratch/klflow_out/logs/kl_6696489_*.log`.

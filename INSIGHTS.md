# Steerable Persons — math/insight track

Doc: https://claude.ai/code/artifact/4b618c37-f9a4-4041-8c37-aa443f2552fc (Proposal A, Proposal B tabs)

## Open research threads (user, 2026-10-14)
- Q1: the method itself as a research question — which representation is "best" for a group of features (compression / causal / identifiable / geometric / behavioral)? → notes_Q1_representation.md
- Q2: compressed sensing → how to encode and steer many features (superposition, coherence, least-interference edits, capacity). → notes_Q2_compressed_steering.md
- Q3: behavioral vs representational geometry (pullback Fisher metric, chart-invariant interaction). → notes_Q3_behavior_representation.md

## Insights log (newest first)
- 2026-10-14 (reviewed, folded into proposals):
  - Unifying object: pullback Fisher metric G = sum_q J_q^T (diag p - pp^T) J_q; H^2 ~ (1/8) d^T G d. Our Hellinger metric is its finite-step form.
  - rho_G (Fisher-weighted interaction) predicts S1-vs-S3 loss (1/8) dgamma^T G dgamma to 2nd order -> Proposal A Fig 3 test vs rho~ and subspace cosine.
  - Plug-in rho is biased toward (I-1)(J-1)/(IJ-1) (~0.71 on 7x5), not up/down uniformly -> cross-fitted rho~ by template halves.
  - ANOVA cannot distinguish linear vs manifold main effects on a finite grid; only held-out values / steering paths can -> nested model selection.
  - Steering = inverse problem: Tikhonov family interpolates matched filter (CAA-like, under causal inner product) <-> pseudo-inverse; leakage bound via coherence; random-frame capacity s* ~ eps^2 d / (2 log N) is a dictionary property, so method gains need structured coherence.
  - Behavioral interaction is chart-dependent (log-odds vs prob).
  - Person side: CS only valid for large-K sparse deviations; dense k<=16 -> Bayesian D-optimal; random-then-adaptive design; 1-bit MC (Davenport 2014).
  - Dropped (wrong/too big): holonomy loop test (additive edits always close; use edit-transfer = 2x2 interaction contrast), direct-sum-manifold identifiability theorem (needs independence + known targets).
  - Main novelty threat: J. M. Lu et al. 2026 (2606.13934) geometry predicts composition errors.
- 2026-10-14: review: S3 (cell-mean steering) is an in-grid oracle; the real test is S3-fit (AMMI) on unseen cells. Additivity (rho=0) != orthogonality. Interaction measured in probabilities vs logits is chart-dependent (to examine in Q3).

## TODO
- Verify volume/issue numbers of classical refs before submission.
- Possible paper-level ideas not in proposals: edit-transfer test; Fisher-orthogonality vs Euclidean predicting leakage; capacity law measurement.

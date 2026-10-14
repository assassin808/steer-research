# Q3 — Behavioral vs representational geometry (multi-concept)

Status legend: [V] = arXiv/abstract page opened 2026-10-14; [V2] = verified via secondary index only; [D] = my derivation (check before citing); [U] = unverified claim.

## 1. Existing frameworks (≤2 lines each)

- **Manifold steering** — Wurgaft, …, Geiger, Lubana, arXiv 2605.05115 [V]. Fit M_h (activations) and M_y (output distributions); steering along M_h traces M_y, linear steering goes off-manifold; reverse optimization recovers M_h curvature. (Hellinger-based M_y per search snippet [U].)
- **Pullback Fisher steering (FishBack)** — Wang & Zhao, arXiv 2605.17231 [V]. Softmax Fisher pulled back via downstream Jacobian = steering metric; claims CAA/ActAdd/ITI implicitly approximate it; GPT-2, three verb-morphology concepts, each steered alone (abstract does not mention joint steering).
- **Riemannian-Manifold Steering** — Oozeer, Raval, …, Abdullah, arXiv 2605.24942 [V]. Steering = geodesics; linear/spline steering = particular metrics; learned encoder approximates the *Hellinger pullback* metric.
- **Curveball steering** — Raval et al., arXiv 2603.09313 [V]. Geodesic/Euclidean distance ratio shows concept-dependent distortion; kernel-PCA nonlinear steering.
- **MISVO** — Entesari, Zhang, Khashabi, Fazlyab, arXiv 2609.30218 [V]. Steering-vector optimization penalized by the local KL (Fisher quadratic) of the induced token distribution.
- **Pulling back information geometry** — Arvanitidis, González-Duque, Pouplin, Kalatzis, Hauberg, AISTATS 2022, arXiv 2106.05367 [V]. Fisher–Rao of the decoder's output distributions pulled back to latent space; the template for G = JᵀFJ.
- **Info-geometric interaction hierarchy** — Amari, IEEE TIT 47(5):1701–1711, 2001 [V2]. Mixed (θ,η) coordinates give Fisher-orthogonal decomposition of k-th order interactions; the classic chart-aware notion of "interaction".
- **Linear representation hypothesis + causal inner product** — Park, Choe, Veitch, ICML 2024, arXiv 2311.03658 [V]; categorical/hierarchical extension: Park, Choe, Jiang, Veitch, arXiv 2406.01506 [V]. Non-Euclidean inner product on h; concept directions estimated from counterfactual pairs.
- **RSA** — Kriegeskorte, Mur, Bandettini, Front. Syst. Neurosci. 2008 [V]; **CKA** — Kornblith et al., ICML 2019, arXiv 1905.00414 [V]; **shape metrics** (Procrustes/CCA as true metrics) — Williams, Kunz, Kornblith, Linderman, NeurIPS 2021, arXiv 2110.14739 [V]. All compare *extrinsic* Euclidean geometry.
- **Metric similarity analysis** — Cayco-Gajic & Pellegrino, arXiv 2603.28764 [V]. Compare *intrinsic* (pullback) geometry instead — closest RSA-style tool to what we need.
- **Causal abstraction** — Geiger et al., arXiv 2301.04709 [V]. Interventional criterion for when a high-level variable (concept) is realized in h; complements geometry with causal faithfulness.
- **Effective dimension** — Maddox, Benton, Wilson, arXiv 2003.02139 [V]: N_eff(G)=Σ λ_i/(λ_i+c); apply to the pullback G to get *behavioral* dimension of a concept subspace.
- **Related results to cite as motivation**: steering vectors are non-identifiable from behavior (Venkatesh & Kurapath, arXiv 2602.06801 [V]) — exactly the kernel of G; near-orthogonal encodings predict successful composition (Lu, …, Saphra, arXiv 2606.13934 [V]; metric used not stated in abstract [U]).

**Honest novelty assessment.** Pullback-Fisher/Hellinger metrics for *single-concept* steering are now crowded (FishBack, MISVO, RMS, Curveball, all 2026). Not found in my searches: (a) multi-concept main-effect/interaction decomposition under the pullback metric; (b) the chart-dependence of behavioral interaction in steering; (c) separating representation-level from readout-level interaction. These are the defensible gaps.

## 2. Formal framework

**Spaces.** Layer-ℓ activation h ∈ ℝ^d (or a fitted M_h ⊂ ℝ^d). Probe set Q; for q∈Q the answer distribution p_q(·|h) ∈ Δ^{K_q−1} obtained by running the network from layer ℓ. Behavior map Φ: h ↦ (p_q(·|h))_{q∈Q} ∈ M_y ⊂ ∏_q Δ^{K_q−1}, with Fisher–Rao metric F = ⊕_q F_q.

**Pullback.** With z_q = f_q(h) the answer logits and J_q = ∂z_q/∂h,
  G(h) = Σ_q J_qᵀ (diag p_q − p_q p_qᵀ) J_q  =  E_{q, y∼p_q}[∇_h log p_q(y|h) ∇_h log p_q(y|h)ᵀ].
rank G ≤ Σ_q (K_q−1) ≪ d: G is a pseudo-metric; ker G = behaviorally null directions (the non-identifiability of 2602.06801). Work on the quotient / column space, or use G_ε = G + εI.

**Link to the current Hellinger measure.** With H²(p,p') = ½Σ(√p−√p')², H²(Φ(h), Φ(h+δ)) = ⅛ δᵀG(h)δ + O(‖δ‖³) [D, standard]. So the proposal's Hellinger metric *is* the pullback metric at second order; the finite-δ Hellinger is the chord of the Fisher–Rao sphere embedding p ↦ 2√p.

**Observation [D, plausibly implicit in FishBack's claim].** At the final layer with p uniform over the vocabulary, G = W_Uᵀ(I/V − 11ᵀ/V²)W_U = Cov_unif(γ), which is the inner product Park et al. induce on representations (ℓ = Cov(γ)^{1/2}λ). The causal inner product is the pullback Fisher metric frozen at the uniform output; G(h) is its state-dependent generalization.

**Behavior-faithful steering.** A curve h(t) is behavior-faithful if it is a geodesic of G restricted to the horizontal space (ker G)^⊥: minimize E[h] = ∫₀¹ ḣᵀG(h)ḣ dt. When Φ is a submersion onto M_y, horizontal G-geodesics are exactly lifts of Fisher–Rao geodesics of M_y (Riemannian submersion) — this is the formal content of "steering along M_h traces M_y". A practical surrogate: pulled-back natural-gradient flow ḣ = −G_ε⁻¹∇_h KL(p* ‖ Φ(h)) (not a geodesic in general; say so).

**Interaction and its chart dependence.** Two binary concepts A,B, class means h_{ab}. Activation interaction Δ_h = h_{11}−h_{10}−h_{01}+h_{00}; its vanishing is invariant to affine charts of ℝ^d, but its *size* and principal angles depend on the metric (Euclidean vs G). In behavior space vanishing is **not** chart-invariant: for any chart ψ of M_y,
  Δ^ψ ≈ Dψ·Δ + D²ψ[δ_A, δ_B],
so additivity in log-odds (natural parameters θ) implies interaction in probabilities (η) and vice versa unless ψ is affine. "No interaction" must be declared relative to a flat structure: α=+1 (e-flat, log-linear, Amari 2001), α=−1 (m-flat, mixtures), or α=0 (√p, Hellinger). Pushing activations through Φ (second-order Taylor at h_{00}, θ-chart):
  Δ_θ ≈ J Δ_h  +  D²θ∘Φ [δ_A, δ_B]                                    (★)
       (representation interaction)   (readout curvature)
(★) [D] splits behavioral interaction into a part *present in the representation* and a part *created by the nonlinear readout* even when activations are perfectly additive.

**Chart-invariant measures.**
1. Fisher cosine cos_G(u,v) = uᵀGv / √(uᵀGu·vᵀGv) and Fisher principal angles between concept subspaces U_A, U_B (compute principal angles of G^{1/2}U_A, G^{1/2}U_B). Invariant to reparametrization of both h and y. cos_G=0 ⇒ Pythagorean first-order behavior: H²(A∧B) ≈ H²(A)+H²(B).
2. α-covariant mixed derivative of ρ(s,t)=Φ(h+su+tv): (∇^{(α)}_{∂_s}∂_t ρ)|₀. Coordinate-free once α is fixed; α=1 recovers mixed partials of log-odds; report the α-profile rather than one chart.
3. Second fundamental form II(∂_s,∂_t) of the surface Φ(h+span{u,v}) in (M_y, F): normal component = intrinsic non-flatness of the composition, independent of chart and of α.
4. Pointwise log-likelihood interaction Λ_y = uᵀ∇²_h log p(y|h) v (note E_y[Λ_y] = −uᵀGv; the variance of Λ_y is extra information).

## 3. Research questions (semester-scale, ≤4B)

Models: Qwen2.5-0.5B/1.5B/3B, Llama-3.2-1B/3B, Gemma-2-2B. Concepts (6–8): sentiment, formality, language (EN/FR), tense, topic, a persona trait. G via JVP/VJP on probe answer logits (|Q|≈50–200, K≤5 ⇒ G rank ≤ 800; store low-rank factor, no d×d matrix).

**RQ1 — Fisher, not Euclidean, orthogonality governs leakage.** Define leakage L_{A→B}(α) = H²(p_B(h+αu_A), p_B(h)) on B-probes. Claim: L_{A→B}(α) = (α²/8)u_Aᵀ G_B u_A + O(α³), hence u_A ⊥_{G_B} (behavioral independence to first order) iff zero small-α leakage. The theory is near-trivial; the empirical questions are: (i) range of α where the quadratic predictor holds; (ii) Spearman of leakage with |cos_E|, |cos_C| (Park), |cos_G| across all ~50 ordered pairs × layers; (iii) does G_B-orthogonalized steering u_A − P^{G}_{B}u_A reduce leakage vs Euclidean Gram–Schmidt at matched target effect. Baselines: CAA, FishBack direction, MISVO-style KL penalty. Falsifiable: if cos_E predicts as well as cos_G, the pullback adds nothing for composition.

**RQ2 — Where does interaction live? (decomposition ★).** For each concept pair and layer, estimate both terms of (★) from class means (JVP for JΔ_h, forward-over-forward for D²Φ[δ_A,δ_B]) in θ, η and √p charts. Hypotheses: (H2a) a large fraction of probability-space ANOVA interaction vanishes in log-odds (scale artifact); (H2b) the residual e-interaction is dominated by readout curvature at late layers and by representation interaction (AMMI rank-1 term) at middle layers. Check Taylor residual; report II-norm as chart-free summary. This directly upgrades the proposal's ANOVA/AMMI from a coordinate statement to a chart-qualified one.

**RQ3 — Composition by geodesic vs linear vs manifold steering.** Target joint behavior p*(a₁,b₁). Compare (i) u_A+u_B, (ii) natural-gradient/horizontal-geodesic path under G_ε, (iii) manifold steering (Wurgaft) on a 2-D grid of concept intensities, (iv) Riemannian-Manifold Steering encoder if code is available [U]. Metrics: target KL, off-target KL on held-out probes, fluency (perplexity under the unsteered model). Prediction: the gap (i)−(ii) is linearly predicted by ‖II‖ / readout-curvature term from RQ2 — linking geometry measurement to steering performance.

Compute: all three are inference + autodiff on ≤3B models; one MIG/A100 slice suffices.

## 4. Suggested Method text

> **Behavioral metric.** For a layer-ℓ representation h, let Φ(h) = (p_q(·|h))_{q∈Q} be the answer distributions on probe questions. We equip behavior space with the Fisher–Rao metric and pull it back, G(h) = Σ_q J_q(h)ᵀ(diag p_q − p_q p_qᵀ)J_q(h), J_q = ∂z_q/∂h (Arvanitidis et al., 2022; cf. Wang & Zhao, 2026). Since H²(Φ(h),Φ(h+δ)) = ⅛δᵀG(h)δ + O(‖δ‖³), our Hellinger behavior distance is the finite-step version of G. We regularize G_ε = G + εI and treat ker G as behaviorally inert.
>
> **Chart-qualified interaction.** For concepts A,B with cell means h_{ab}, we decompose behavioral interaction in natural-parameter (log-odds) coordinates as Δ_θ ≈ J Δ_h + ∇²(θ∘Φ)[δ_A,δ_B], separating representation-level interaction (Δ_h, modeled by AMMI) from readout curvature. Because vanishing interaction is chart-dependent in behavior space, we report it under the e-, m- and Hellinger charts and summarize it chart-free by the second fundamental form of Φ(h_{00}+span{δ_A,δ_B}).
>
> **Fisher geometry of concept subspaces.** Principal angles between concept subspaces are computed both in the Euclidean metric and in G (angles between G^{1/2}U_A and G^{1/2}U_B); behavior-faithful composition is defined as a horizontal geodesic of G, approximated by pulled-back natural-gradient flow toward the target joint behavior.

## Sources
- https://arxiv.org/abs/2605.05115 · https://arxiv.org/abs/2605.17231 · https://arxiv.org/abs/2605.24942 · https://arxiv.org/abs/2603.09313 · https://arxiv.org/abs/2609.30218
- https://arxiv.org/abs/2106.05367 · https://arxiv.org/abs/2311.03658 · https://arxiv.org/abs/2406.01506 · https://arxiv.org/abs/2110.14739 · https://arxiv.org/abs/1905.00414
- https://arxiv.org/abs/2603.28764 · https://arxiv.org/abs/2301.04709 · https://arxiv.org/abs/2003.02139 · https://arxiv.org/abs/2602.06801 · https://arxiv.org/abs/2606.13934
- RSA: https://pubmed.ncbi.nlm.nih.gov/19104670/ · Amari 2001 (via https://arxiv.org/abs/1601.05533 description) [V2]

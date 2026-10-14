# Adversarial review of notes Q1–Q3 + edit plan for Proposals A and B

Date: 2026-10-14. Reviewer pass: all arXiv IDs cited in the three notes were resolved (arXiv API or abs page); the novelty-bearing ones were read at abstract level, and SAE-TS appendix A.1 was read in full text. The Docs proposal was read (A rev 32, B rev 24) and not edited.

---

## Part 1. Citation check

### 1.1 Papers that carry a novelty claim

| ID | Exists | Says what notes claim? | Issue |
|---|---|---|---|
| 2605.17231 FishBack (Wang, Zhao, Cao, Yao, Shu) | yes | Yes: softmax Fisher pulled back through intervening-layer Jacobian; says CAA/ActAdd/ITI assume a Euclidean activation space (each implicitly uses an approximate metric); closed-form direction. No joint steering is mentioned. | **Misdescription (minor):** Q3 says "GPT-2" only. The abstract page lists GPT-2 Small, Llama-3-8B and Qwen3-8B. The three verb-morphology concepts are correct. |
| 2609.30218 MISVO (Entesari, Zhang, Khashabi, Fazlyab) | yes | Yes: steering penalized by local KL (Fisher quadratic) of the token distribution. | It is **pre-logit** steering toward a test-time reward (single objective), on 1B–14B models. Not concept steering. Cite only as "KL-penalized steering". |
| 2605.24942 Riemannian-Manifold Steering (Oozeer, Raval, …, Abdullah) | yes | Yes: steering as geodesics; linear and spline steering are special metrics; output-space Hellinger pulled back to activations, approximated by a learned encoder. | Fine. Single-concept arithmetic benchmark. |
| 2603.09313 Curveball (Raval, Song, Wu, Harrasse, Phillips, Barez, Abdullah) | yes | Yes: geodesic/Euclidean ratio shows concept-dependent distortion; polynomial kernel-PCA steering. | Fine. |
| 2603.28764 (Cayco-Gajic & Pellegrino) | yes | Partly. Real title "Geometry-aware similarity metrics for neural representations on Riemannian and statistical manifolds" (MSA). Compares intrinsic vs extrinsic geometry. | The abstract does not say "pullback". Do not describe it as a pullback-metric method without reading the paper. |
| 2602.06801 (Venkatesh & Kurapath) | yes | Yes: steering vectors non-identifiable from behaviour. | Scoped to white-box single-layer access; empirical (orthogonal perturbations work almost as well). Linking it to ker G is our interpretation, not their claim. |
| 2608.27540 (Ivanitskiy, Jasper, King, Mixon) | yes | Mostly. Encoding Wx, readout ReLU(WᵀWx+b); support recovery up to expected sparsity of order d/log n for nearly tight low-coherence dictionaries; exact worst-case threshold in terms of coherence for real ETFs with n>d+1. | Notes say ETFs "attain" the threshold. Correct wording: the threshold is *determined for* ETFs. Q2 says the capacity conjecture "matches" this result. It shares the d/log N **scaling**, but the problem is different (readout leakage under editing vs support recovery). |
| 2411.02193 SAE-TS (Chalnev, Siu, Conmy) | yes | **Confirmed in full text.** App. A.1, Table 3: the pseudo-inverse of the effect matrix M scores far below SAE-TS (e.g. London 0.538 vs 0.072); Table 6 does the same for behavioural×coherence. | The notes explain the gap by noise amplification (large κ). That does not fit the setup: SAE-TS solves for a full target effect vector over **all** SAE features (N≫d). This is the over-determined regime that P3 itself calls infeasible. The failure is more plausibly forced over-determination than conditioning. Say both. |
| 2408.05451 (Hänni, Mendel, Vaintrob, Chan) | yes | Yes: pairwise ANDs with Õ(m^{2/3}) neurons; width d emulates circuits of width Õ(d^{1.5}), with ε-error. | Fine. |
| 2211.16467 Squires, Seigal, Bhate, Uhler | yes | Yes: linear mixing of a linear latent SCM; one intervention per latent is sufficient, and missing one breaks identifiability. | The notes claim Q1 §3.2 "generalizes Squires". That is an overclaim (see 2.1.2). |
| 2006.12982 GEOMANCER (Pfau, Higgins, Botev, Racanière) | yes | Yes: approximates the de Rham decomposition; unsupervised factorization possible if the metric is known and each factor has nontrivial holonomy. | Fine. |
| 2606.13934 (J. M. Lu, …, Pavlick, Saphra) | yes | "Adversarial Concept Search": near-orthogonal concept encodings predict successful composition. | **Novelty threat for Proposal A, currently not cited.** It already predicts composition failure from pairwise geometry. A must cite it and state the difference (behavioural metric, interaction term, steering rather than prompting). Note the name clash with "Lu et al., 2026" (Assistant Axis, C. Lu) already in A. |
| 2605.05115 Wurgaft et al. | yes | Yes. | The "Hellinger M_y" detail is still unverified [U]; A does not rely on it. |

### 1.2 Other citations (existence + title)

All exist with the claimed authors: 2003.12298, 2002.10689, 2301.04709, 2303.02536, 2311.17030, 2505.24535 (K-Steering, Oozeer et al.), 2508.10599 (MSRS, Jiang et al.), 2603.03326 (Hoppe et al.; real title "Controllable and explainable personality sliders…", method SAS, Big Five; matches), 1811.12359, 1907.04809, 2306.00542, 2007.00810, 2410.23501, 2402.09236, 2311.03658, 2406.01506, 2405.14860, 2502.00873, 2505.18235, 2408.10920, 2407.14662, 1812.02230, 2209.10652, 2503.01824, 0906.3173, 1206.5882, 1308.6273, 2403.05767, 2406.17563, 2502.12446 (MAT-Steer: sparsity + orthogonality among attribute vectors; matches), 2306.03819, 1202.1212, 1111.4646, 1001.5311, 2106.05367, 2110.14739, 1905.00414, 2003.02139. **No fabricated IDs found.**

### 1.3 Missing prior work that should be cited
- **Conceptors for Semantic Steering** (Triantafyllopoulos et al., arXiv 2605.04980): closed-form Boolean composition of concept subspaces; beats additive baselines where subspaces are multidimensional. This is a composition baseline for A.
- **1-bit matrix completion** (Davenport, Plan, van den Berg, Wootters, arXiv 1209.3672). The binary/ordinal person×item matrix is exactly its setting. It fits B better than Candès–Recht (real-valued entries).

### 1.4 Novelty status after search
- "Multi-concept interaction under the pullback-Fisher metric": not found. FishBack, MISVO, RMS and Curveball are single-concept, and conceptors are subspace-based. **Defensible**, but Lu et al. 2026 (geometry predicts composition) narrows it. The differentiator must be (i) the behavioural metric and (ii) the interaction term γ, not orthogonality alone.
- "Compressed sensing / coherence theory of simultaneous steering capacity": no LLM-steering hits. Defensible as a conjecture plus measurement.
- "RIP / compressed sensing for MIRT item selection": no hits. It is defensible only in a regime the proposal does not currently use (see 2.2.4).

---

## Part 2. Math check

### 2.1 Q1 (representation)

**2.1.1 Fisher-weighted interaction ρ_G (§3.1).**
- *The second-order expansion is correct.* The additive edit lands at μ_{i'j} − Δγ, so KL ≈ ½ΔγᵀGΔγ. For a fixed Ḡ and sum-to-zero γ, averaging over all (i,i') gives Σ_{i,i'}ΔγᵀḠΔγ = 2I·Σ_iγ_{ij}ᵀḠγ_{ij}, so ρ_G is the right summary up to constants.
- **Gap 1 (Jensen / context noise).** Steering acts per context, not on the mean. Write h_{c,ij} = μ_ij + e_{c,ij}. The per-context error is −Δγ + (e_{c,ij} − e_{c,i'j}), and the second part is shared by S1 and S3. The clean prediction is therefore about the **gap** between S1 and S3 (or S3-fit), not the absolute error:
  E_c KL(S1) − E_c KL(S3) ≈ ½ΔγᵀḠΔγ, with Ḡ = E_c G(h_c), assuming the two error parts are uncorrelated. This maps directly onto A's Figure 3.
- **Gap 2 (which G).** ρ_G is undefined until G is fixed. Use Ḡ, the cell- and context-averaged pullback; report the variation across cells as a diagnostic.
- **Error (MDL side).** MDL under a Gaussian noise model scores γ against the noise covariance Σ_ε (Mahalanobis), not the Euclidean norm. The correct contrast is ρ_{Σ⁻¹} (detectability) versus ρ_G (steerability), not ρ_E versus ρ_G.
- **Overclaim.** "The criteria disagree *exactly* when γ concentrates in low/high-Fisher directions" is a restatement of ρ_Σ ≠ ρ_G, not a theorem. Present it as a measurable hypothesis: ρ_G predicts the S1-vs-S3 gap better than ρ̃.

**2.1.2 Identifiability of direct-sum manifold codes (§3.2). As stated, it is false.**
- With "arbitrary dependence among z_k", changing p(z_k) generally changes p(z_{−k}) (descendants) and the cross-covariances Cov(E_k, E_l). The cross block has column space in S_l, so col(ΔCov h) ⊄ S_k.
- Minimal fix: assume the intervention leaves p(z_{−k}) and all cross-covariances unchanged; for example z_k ⟂ z_{−k} in both regimes. The mean shifts then also lie in S_k.
- The "generic: col(ΔCov) = S_k" assumption nearly assumes the conclusion. The result reduces to elementary linear algebra (independent-subspace analysis with known targets).
- "Affine hulls form a direct sum" should be "the linear spans of the centred images E_k(M_k) − E_k(z_0) form a direct sum".
- **Overclaims.** (i) It does not generalize Squires et al., which identifies a dependent linear SCM. This result needs independence and **known** intervention targets. (ii) "Applies when concept labels are latent" is wrong, because the single-trait prompts *are* the labels.
- Recommendation: **do not put §3.2 in the proposal.** At most keep "single-trait interventions + oblique projection" as an exploratory diagnostic.

**2.1.3 Split-half ρ (§3.3 i) and A's current noise correction.**
- **Error in the notes:** "plug-in ρ̂ is biased upward". The numerator is inflated by (I−1)(J−1)σ²/n and the denominator by (IJ−1)σ²/n, where σ² = tr Σ_ε. So the plug-in ρ̂ is biased **toward (I−1)(J−1)/(IJ−1)**, not always upward. For 7×5 that value is 24/34 ≈ 0.71, so at low SNR every pair looks strongly entangled. Strongly entangled pairs can be biased downward. Also, for 2×2 the interaction has the *fewest* degrees of freedom (1 vs 2).
- **Error in Proposal A (§3.2):** ρ̃ subtracts the noise term from the numerator only. The denominator also needs the (IJ−1)·tr(Σ̂_ε)/n correction.
- The cross-fitted estimator ⟨γ̂^A, γ̂^B⟩ / ⟨μ̂^A − m̂^A, μ̂^B − m̂^B⟩ has unbiased numerator and denominator if the halves are independent. The ratio is only consistent, and it can be negative.
- **The split must be by template/nuisance c, not by prompt**, because A's design crosses c with all cells and c×cell effects are correlated within a template.
- The cross-fitted version is preferable because it needs no homoscedasticity or equal-n assumption.

**2.1.4 Holonomy loop (§3.3 iii). Mathematically wrong.**
- Additive edits are state-independent constant vectors: (α_{i'}−α_i) + (β_{j'}−β_j) + (α_i−α_{i'}) + (β_j−β_{j'}) = 0. So the loop closes for **every** representation, whatever γ is. Cell-mean edits (S3) also telescope to zero.
- Holonomy is nontrivial only for state-dependent steering fields (manifold or natural-gradient steering).
- The correct test with this content is **edit transfer**. Apply the edit measured at b_j, μ_{i'j} − μ_{ij}, in context b_{j'}. The landing error is exactly the 2×2 interaction contrast:

```latex
\big(\mu_{ij'} + \mu_{i'j} - \mu_{ij}\big) - \mu_{i'j'} = -\big(\gamma_{i'j'} - \gamma_{i'j} - \gamma_{ij'} + \gamma_{ij}\big).
```

  Scoring it in output space gives ½cᵀGc with c the contrast. Rename the diagnostic accordingly.

**2.1.5 Class chain.** The notes write R_add ⊂ R_⊕M. A parametric chart *constrains* the main effects, so the order is R_⊕M ⊆ R_add ⊂ R_AMMI(r) ⊂ R_full. The caveat that charts are detectable only on held-out values is correct and important.

### 2.2 Q2 (compressed steering)

- **P1: correct.** argmin‖Δh‖_M s.t. R_TΔh = δ is M⁻¹R_Tᵀ(R_TM⁻¹R_Tᵀ)⁻¹δ, and the leakage is G_{jT}G_{TT}⁻¹δ. The Gershgorin bound √sμ‖δ‖/(1−(s−1)μ) is correct when (s−1)μ < 1.
- **P2: algebra correct.** The target error is (G_TT−I)δ with ‖·‖_∞ ≤ (s−1)μ‖δ‖_∞. CAA = P1 iff G_TT = I. The Tikhonov limits are right.
- **Modelling caveat (important).** P2 treats the readout rows r_i as the steering (write) directions. In models, read ≠ write: an SAE encoder differs from its decoder, and difference-of-means is a write direction whose optimal probe is Σ⁻¹Δμ. "CAA = λ→∞ limit" holds only if difference-of-means ∝ M⁻¹rᵢ, which is the causal-inner-product condition of Park et al. Using **behavioural readouts** (probe-logit Jacobians J_q) avoids the issue, because the edit is in h and the readout is its linearized effect. This is the formulation recommended below.
- The "optimal λ ≈ σ²s" borrows MMSE precoding, but there the noise is additive channel noise. Here it is error in the estimated directions (errors-in-variables). Label it a heuristic and tune λ on held-out data.
- **Method paragraph error.** With M ≠ I the solution is M⁻¹R_Tᵀ(G_TT+λI)⁻¹δ, with G = RM⁻¹Rᵀ. The paragraph drops the M⁻¹. Check: (λM + R_TᵀR_T)⁻¹R_Tᵀ = M⁻¹R_Tᵀ(λI + R_TM⁻¹R_Tᵀ)⁻¹ by the push-through identity.
- **P3: energy formula correct.** But the bound 1/(1−(|S|−1)μ) is vacuous once |S| ≳ 1/μ ≳ √d (Welch). "|S| ≤ d" is only feasibility. "Top-k neighbours is the greedy optimum" is unproven; call it a heuristic. "N > d ⇒ zero leakage everywhere impossible" holds generically.
- **P4: correct** (conditional Gaussianity, the Davidson–Szarek bound on σ_min(R_T), a union bound, and a matching lower bound). The (1+o(1)) needs s/d → 0. The rows are N(0, I/d), so they are only approximately unit-norm.
- **Missed corollary (useful).** Under the random frame, CAA's leakage G_{jT}δ is also ~ N(0, ‖δ‖²/d). So **CAA and the min-norm edit have the same leakage order on random dictionaries**; they differ only in target error. Hence: (i) the capacity law is a property of the dictionary, not of the method; (ii) any advantage of pseudo-inverse or Tikhonov over CAA must come from **structured** coherence (heavy tail of |G_ij|). This gives a sharper, falsifiable version of C1.
- Also: P1 minimizes ‖Δh‖, and E_j|r_jᵀΔh|² = ‖Δh‖²/d for isotropic unknown non-targets. So "minimum norm = least expected interference" is a correct justification and worth stating.
- **Capacity: the algebra is right.** With ‖δ‖² = sτ², s* = ε²d/(2 log N). Example: d = 2048, N = 16k gives s* ≈ 100ε², so about 9 targets at ε = 0.3. This is a concrete, testable number. The prediction concerns readout leakage at the layer, not behaviour; say so.
- **P5.** A tangent step of length L deviates by ≤ κL²/2 ≤ L²/(2τ). That is correct as an upper bound. But the difference-of-means edit between two values is a **chord**, which lands on the manifold. Off-manifold error arises when one global direction is reused for several moves or when intermediate points are traversed. Reword.
- **§4, person side.**
  - RIP needs random (sub-Gaussian) design, but item loadings are fixed by the bank; only rows can be selected. Guarantees do not transfer, so the result is a heuristic.
  - "D-optimality fails when n < K (singular Fisher)" is wrong for the **Bayesian** criterion det(Σ⁻¹ + ΣI_j) that B already uses. The real issue is that a Gaussian prior is not sparsity-adapted.
  - The Arias-Castro–Candès–Davenport "adaptivity ≤ log factors" result is for linear measurements under an energy budget. Logistic items have θ-dependent information, so adaptive testing can gain more than that. Do not use the result to discount adaptive selection; use it as a regime test instead.
  - The 1-bit CS sample complexity (Plan–Vershynin) has a poor polynomial dependence on accuracy and needs Gaussian measurements. "s log(K/s) bits" is the scaling only.

### 2.3 Q3 (behaviour vs representation)

- **G formula: correct**, including rank ≤ Σ(K_q−1) and the information identity E_y[∇²_h log p] = −G. Λ_y is fine.
- **H² ≈ ⅛δᵀGδ: correct** for H² = ½Σ(√p−√p')²: Σ(ṗ/2√p)² = ¼I, then ×½. The proposal's D_H is the square root of this, ≈ ‖δ‖_G/(2√2).
- **Final-layer uniform observation: correct.** Park et al.'s inner product on the representation side is Cov(γ), and G at uniform p equals Cov_unif(γ). Probably implicit in FishBack's "each method adopts an approximate metric"; do not claim it as new.
- **Riemannian-submersion statement.** Only approximately true. With G_ε = G + εI, G_ε restricted to the horizontal space is not the pullback; equality needs ε → 0 and constant rank of dΦ. Say "in the limit ε→0, under constant rank".
- **(★) chart decomposition: correct to second order**, with the same Jensen caveat as 2.1.1. The class-mean behaviour p_ab is E_c Φ(h_c), not Φ(E_c h_c). Either evaluate Φ at patched mean activations or apply (★) per context and average. The Hessian term needs forward-over-reverse autodiff. That is feasible at ≤ 3B but not cheap; put it in an ablation.
- **Error in RQ1 (Fisher orthogonality).** The leakage of edit u_A onto B's probe is ⅛α²·u_AᵀG_Bu_A: a **G_B-norm of u_A**, not an angle between u_A and u_B. Zero small-α leakage holds iff G_Bu_A = 0, i.e. u_A ⊥ rowspace(J_B). The Fisher cosine cos_G(u_A, u_B) = 0 instead gives additivity of the *total* divergence, H²(u+v) ≈ H²(u) + H²(v). These are two different claims and RQ1 conflates them. The "G_B-orthogonalized" edit should be the projection of u_A onto ker J_B, which is exactly Q2's protected-set constraint with R_P = J_B. **This is the natural bridge between Q2 and Q3.**

---

## Part 3. Synthesis: one minimal set of additions

**Unifying object.** The behavioural readout map is J_q(h) = ∂z_q/∂h, with pullback metric G = Σ_qJ_qᵀ(diag p_q − p_qp_qᵀ)J_q.
- **Q1 → interaction measured in G.** ρ_G predicts the S1-vs-S3 gap; it is a second-order identity, not a conjecture.
- **Q3 → leakage in H² is ⅛ΔhᵀG_BΔh.** Composition is additive when cos_G = 0. Interaction is chart-dependent on the behaviour side.
- **Q2 → least-interference steering** is min-norm inversion of the stacked J's (targets) with protected rows (other concepts). Tikhonov interpolates between a matched-filter/CAA-like edit and the pseudo-inverse. On random dictionaries the leakage order is ‖δ‖√(2 log N/d), which gives a capacity conjecture.

**Drop:** the identifiability theorem (§3.2 of Q1), the holonomy test (replace it with edit transfer), the full Pareto/MDL machinery (too big for 2–4 pages; keep only held-out model selection), the second fundamental form and α-connections (keep one sentence on chart dependence), and the specific claim about optimal λ.

---

## Part 4. Edit plan, Proposal A (node 750219a9-55e6)

Net length change is about +0.4 page. Several weaker sentences are replaced.

### A1. Replace the RQ1 text
**Find:** "For two concepts, how much of the variation in mean activations is explained by separate per-concept terms, and how much by their interaction? Are the per-concept subspaces orthogonal?"
**Replace with:**
> Which model of the joint representation of two concepts (additive, additive with per-concept manifold charts, additive plus low-rank interaction, or unconstrained) best predicts held-out cells, and how large is the interaction when it is measured in the metric that the model's own outputs induce rather than in the Euclidean metric?

### A2. Replace contribution (1)
**Find:** "(1) A measurement protocol for the joint structure of two concepts, based on a standard two-way decomposition of class-conditional means."
**Replace with:**
> (1) A measurement protocol for the joint structure of two concepts: held-out selection among nested representation models, and a behavioural (pullback-Fisher) interaction fraction that predicts, to second order, how much additive steering loses.

### A3. Replace the Method lead sentence
**Find:** "All tools are standard; the novelty is the controlled comparison."
**Replace with:**
> The individual tools are standard. What is new is selecting the representation model on held-out cells, weighting the interaction by its effect on behaviour, and a least-interference steering family that contains additive steering as a limit.

### A4. Replace the "Superposition" related-work paragraph
**Find:** the paragraph starting "Superposition (Elhage et al., 2022). Models represent more features than dimensions…" and ending "…interaction terms we measure."
**Replace with:**
> **Geometry-aware and multi-concept steering.** Pullback-Fisher and pullback-Hellinger metrics give principled single-concept steering directions and paths (Wang et al., 2026; Oozeer et al., 2026), following the pullback of information geometry to latent spaces (Arvanitidis et al., 2022). For several attributes, orthogonality penalties, sequential orthogonalization and conceptor algebra reduce interference (Nguyen et al., 2025; Hoppe et al., 2026; Triantafyllopoulos et al., 2026). Pairwise encoding geometry predicts compositional failures in prompting (J. M. Lu et al., 2026), and superposition is analysed as sparse recovery from an overcomplete frame (Elhage et al., 2022; Ivanitskiy et al., 2026). *None of these measures the interaction term itself, or weights it by its effect on behaviour. We do both, and we relate steering leakage to the coherence of the concept dictionary.*

### A5. Insert model selection in §3.2 after the "Per-concept geometry" list item
**Insert after:** "…fitted with the manifold-fitting procedure of Wurgaft et al. (2026)."
> **Model selection.** The candidate models of the cell means are nested,

```latex
\mathcal R_{\oplus\mathcal M} \subseteq \mathcal R_{\mathrm{add}} \subset \mathcal R_{\mathrm{AMMI}}(R) \subset \mathcal R_{\mathrm{full}},
```

> where 𝓡_add is the additive model m + α_i + β_j, 𝓡_⊕𝓜 constrains the main effects to parametric charts α_i = f_A(θ_i), β_j = f_B(φ_j) (circle, helix, line), 𝓡_AMMI(R) adds the rank-R interaction, and 𝓡_full has one free mean per cell. In-sample fit is monotone along this chain, so models are compared only out of sample: by leave-one-cell-out error for R, and by leave-one-value-out error for the charts. On a finite grid a chart cannot be distinguished from free main effects without held-out values.

### A6. Replace the noise-correction sentence in §3.2 (fixes the denominator bug)
**Find:** "Because sampling noise inflates the plug-in interaction sum of squares by (I − 1)(J − 1)·tr(Σ_ε)/n, with n contexts per cell, we report the noise-corrected ρ̃ that subtracts this term"
**Replace with:**
> Sampling noise inflates the plug-in numerator by (I − 1)(J − 1)·tr(Σ_ε)/n and the denominator by (IJ − 1)·tr(Σ_ε)/n. The plug-in ρ is therefore biased toward (I − 1)(J − 1)/(IJ − 1), not toward zero. We report instead the cross-fitted estimate computed from two disjoint halves of the templates,

```latex
\tilde\rho_{AB} = \frac{\sum_{ij}\langle \hat\gamma^{(1)}_{ij},\, \hat\gamma^{(2)}_{ij}\rangle}{\sum_{ij}\langle \hat\mu^{(1)}_{ij}-\hat m^{(1)},\, \hat\mu^{(2)}_{ij}-\hat m^{(2)}\rangle},
```

> whose numerator and denominator are unbiased when the template halves are independent

Keep the existing ending (". The same decomposition…"), unless A7 is applied too.

### A7. Replace the behaviour-side sentence in §3.2 with behavioural geometry
**Find:** "The same decomposition is applied on the behavior side to the log-probabilities of the probe answers, so structure in activations can be compared with structure in behavior."
**Replace with:**
> **Behavioural metric.** Let z_q(h) be the probe-answer logits obtained by running the model forward from an edited h_l, with p_q = softmax(z_q) and J_q = ∂z_q/∂h. The Fisher–Rao metric of the answer distributions pulled back to activations is

```latex
G(h) = \sum_{q\in\{A,B\}} J_q(h)^{\top}\big(\operatorname{diag} p_q - p_q p_q^{\top}\big) J_q(h),
\qquad
H^2\big(p(\cdot\mid h),\, p(\cdot\mid h+\delta)\big) = \tfrac18\,\delta^{\top} G(h)\,\delta + O(\lVert\delta\rVert^3),
```

> so the Hellinger distance in §3.4 is the finite-step form of G. With Ḡ the average of G over cells and contexts, an additive edit (a_i, b_j) → (a_i′, b_j′) misses the counterfactual mean by Δγ = γ_i′j′ − γ_ij. To second order it loses ⅛ ΔγᵀḠΔγ in squared Hellinger distance relative to S3. This motivates the behavioural interaction fraction

```latex
\rho^{G}_{AB} = \frac{\sum_{ij}\gamma_{ij}^{\top}\bar G\,\gamma_{ij}}{\sum_{ij}(\mu_{ij}-m)^{\top}\bar G\,(\mu_{ij}-m)},
```

> which is small when the interaction lies in directions the outputs ignore (ker Ḡ), even if ρ̃ is large. On the behaviour side we apply the same decomposition to probe log-probabilities. Vanishing interaction there depends on the chart (log-odds versus probabilities), so we report both.

### A8. Add one row to the §3.3 table and one paragraph after the falsifiable prediction
**New table row (after S3-fit):** `S4 Least-interference` | `M⁻¹R_Tᵀ(R_T M⁻¹R_Tᵀ + λI)⁻¹ δ, with R_T the linearized readouts of the changed concepts and protected readouts constrained to zero` | `main effects plus the readout geometry of all concepts`

**Insert after:** "…Steering strength and layer are chosen on training templates."
> **Least-interference steering.** Stack the linearized readouts of the concepts to be changed as R_T (rows of J_q, or linear probes) and those to be held fixed as R_P. We solve

```latex
\Delta h_\lambda = \arg\min_{\Delta h}\; \lVert\Delta h\rVert_M^2 + \lambda^{-1}\lVert R_T\Delta h - \delta\rVert^2
\quad \text{s.t. } R_P\,\Delta h = 0,
\qquad
\Delta h_\lambda\big|_{R_P=\varnothing} = M^{-1}R_T^{\top}\big(R_T M^{-1}R_T^{\top} + \lambda I\big)^{-1}\delta .
```

> The limit λ → 0 is the minimum-norm (pseudo-inverse) edit, which hits the targets exactly. The limit λ → ∞ gives, up to scale, the matched-filter edit M⁻¹R_Tᵀδ. That edit coincides with summed difference-of-means vectors when the latter are dual to the readouts, as under the causal inner product (Park et al., 2023). With M = I, the leakage onto an unconstrained concept j is r_jᵀΔh_0 = G_jT G_TT⁻¹δ, where G = RRᵀ has unit diagonal and coherence μ = max_{i≠j}|G_ij|. Hence

```latex
\lvert r_j^{\top}\Delta h_0\rvert \le \frac{\sqrt{s}\,\mu}{1-(s-1)\mu}\,\lVert\delta\rVert_2, \qquad s = |T|,\ (s-1)\mu<1 .
```

> The protected-set constraint R_PΔh = 0 with R_P = J_B removes the first-order leakage ⅛ΔhᵀG_BΔh onto concept B. A naive pseudo-inverse that constrains all features at once is over-determined and underperforms in practice (Chalnev et al., 2024). We tune λ on training templates.

### A9. Change the Figure 3 caption and the E3 expected outcome
**Find:** "against ρ̃ (x), one point per concept pair and layer."
**Replace with:** "against ρ̃ (left) and ρ^G (right), one point per concept pair and layer; the dashed line is the second-order prediction ⅛ΔγᵀḠΔγ."
**Find (in E3 Expected outcome):** "with an advantage that increases in ρ̃ (positive slope in Figure 3)."
**Replace with:** "with an advantage that increases in ρ̃ and is predicted better by ρ^G than by ρ̃ (higher rank correlation in Figure 3)."

### A10. Replace the E4 three-concept bullet
**Find:** "Three concepts at once on the synthetic task (weekday × month × count), using the three-way decomposition."
**Replace with:**
> Many concepts at once: s ∈ {2, 4, 8, 16, 32} persona or SAE directions steered jointly with S1 and S4 (and the three-way decomposition for weekday × month × count). If the readouts behaved like a random frame, the largest leakage onto unsteered concepts would scale as ‖δ‖√(2 log N / d), and joint steering with relative leakage at most ε would be possible up to s* ≈ ε²d/(2 log N). We test this scaling against the empirical Gram matrix. Under the random-frame model S1 and S4 have the same leakage order, so any advantage of S4 must come from structured, clustered coherence.

### A11. References to add
- Arvanitidis, G., González-Duque, M., Pouplin, A., Kalatzis, D., Hauberg, S. (2022). Pulling back information geometry. AISTATS. arXiv 2106.05367.
- Chalnev, S., Siu, M., Conmy, A. (2024). Improving Steering Vectors by Targeting Sparse Autoencoder Features. arXiv 2411.02193.
- Hoppe, F., Khachaturov, D., Mullins, R., Meng, M. H. (2026). Controllable and explainable personality sliders for LLMs at inference time. arXiv 2603.03326.
- Ivanitskiy, M. I., Jasper, J., King, E. J., Mixon, D. G. (2026). Towards a mathematical theory of superposition. arXiv 2608.27540.
- Lu, J. M., Zhang, R., Lee, I., Alvarez-Melis, D., Pavlick, E., Saphra, N. (2026). Adversarial Concept Search: Predicting Compositional Errors From Feature Geometry. arXiv 2606.13934.
- Nguyen, D., Prasad, A., Stengel-Eskin, E., Bansal, M. (2025). Multi-Attribute Steering of Language Models via Targeted Intervention. arXiv 2502.12446.
- Oozeer, N., Raval, S., et al. (2026). Riemannian-Manifold Steering. arXiv 2605.24942.
- Triantafyllopoulos, I., et al. (2026). Conceptors for Semantic Steering. arXiv 2605.04980.
- Wang, S., Zhao, J., Cao, Q., Yao, H., Shu, L. (2026). FishBack: Pullback Fisher Geometry for Optimal Activation Steering in Transformers. arXiv 2605.17231.
- Already listed in the to-do: cite Kantamneni & Tegmark in the E1 text.

---

## Part 5. Edit plan, Proposal B (node 9105138b-a98c)

Compressed sensing appears only where it is valid. Net length change is about +0.2 page.

### B1. Replace the italic sentence in the CS related-work paragraph
**Find:** "It gives the conceptual frame for RQ1: a person is the signal, an answered item is a measurement, and the item bank is the measurement matrix."
**Replace with:**
> *It frames RQ1, with the person as the signal and the item bank as the measurement matrix. Its guarantees apply only when the number of trait dimensions K exceeds the number of answers and a person's deviation from the population is sparse. For a few dense traits, Bayesian optimal design is the right tool. Binary and ordinal answers are covered by 1-bit compressed sensing and 1-bit matrix completion (Plan and Vershynin, 2013; Davenport et al., 2014).*

### B2. Replace the last sentence of "Measurement view"
**Find:** "Across people the person-by-item matrix is approximately rank k, which links RQ1 to low-rank matrix completion (Candès and Recht, 2009) and, when traits are sparse in an overcomplete dictionary, to compressed sensing (Candès and Tao, 2006)."
**Replace with:**
> Across people the person-by-item matrix of answer probabilities is approximately rank k, which links RQ1 to 1-bit matrix completion (Davenport et al., 2014). We also test a sparse-deviation regime with many fine-grained facets, K ≫ |S|, and θ_n = θ̄ + s_n with ‖s_n‖_0 ≤ s. If the loadings A_S satisfied the restricted isometry property, about

```latex
|S| \gtrsim C\, s \log(K/s)
```

> answers would recover s_n uniformly over supports (Candès and Tao, 2006), and a comparable number of binary answers would suffice up to accuracy-dependent constants (Plan and Vershynin, 2013). Item banks are fixed rather than random, so we treat this bound as a design heuristic and test it empirically. The Gaussian prior behind D-optimality is not adapted to sparse deviations.

### B3. Add a fourth item-selection rule to the §3.2 list
**Insert as item 4 after:** "…the Bayesian D-optimality criterion for multidimensional adaptive testing (Segall, 1996)."
> **Random-then-adaptive** selection: the first n₀ ≈ s log(K/s) items are drawn at random, to discover which traits deviate from the population under a sparse (Laplace) prior. The remaining items are chosen by D-optimality on the estimated support. For linear measurements, adaptivity improves sparse estimation by at most a logarithmic factor (Arias-Castro et al., 2013). A large gain of adaptive over random selection at small K therefore indicates that the nonlinearity of the response model, not sparsity, drives the benefit.

### B4. Extend E1 (Figure 1 caption and expected outcome)
**Find:** "one line each for random, greedy and adaptive selection;"
**Replace with:** "one line each for random, greedy, adaptive and random-then-adaptive selection, for the dense model (k ≤ 16) and the sparse-deviation model (K = 64);"
**Append to the E1 Expected outcome:** "Random-then-adaptive helps only in the sparse-deviation model; if it does not help there, compressed sensing adds nothing to item selection on this panel."

### B5. References to add
- Arias-Castro, E., Candès, E. J., Davenport, M. A. (2013). On the fundamental limits of adaptive sensing. *IEEE Transactions on Information Theory*, 59 (2013). arXiv 1111.4646.
- Bock, R. D., Aitkin, M. (1981). Marginal maximum likelihood estimation of item parameters. *Psychometrika*, 46(4). (pending item from INSIGHTS)
- Candès, E. J., Recht, B. (2009). Exact matrix completion via convex optimization. *Foundations of Computational Mathematics*, 9(6). (keep if B2 still cites it; otherwise drop)
- Davenport, M. A., Plan, Y., van den Berg, E., Wootters, M. (2014). 1-bit matrix completion. *Information and Inference*, 3(3). arXiv 1209.3672.
- Plan, Y., Vershynin, R. (2013). Robust 1-bit compressed sensing and sparse logistic regression: a convex programming approach. *IEEE Transactions on Information Theory*, 59(1). arXiv 1202.1212.
- Segall, D. O. (1996). Multidimensional adaptive testing. *Psychometrika*, 61(2). (pending item from INSIGHTS)

Check the volume/issue numbers before submission: they are given from memory and are not confirmed on the arXiv pages.

---

## Part 6. Open risks for the proposals
1. **J. M. Lu et al. 2026** is the closest competitor to A's thesis. If the course reviewers know it, A must show that the interaction term and ρ^G add predictive power beyond pairwise orthogonality. Add "|cos| between U_A and U_B" as a baseline predictor in Figure 3.
2. Estimating G needs a JVP/VJP through the remaining layers for each context, which is cheap at ≤ 4B (rank ≤ Σ(K_q−1)). The Hessian term of (★) is not proposed; keep it out.
3. A8's S4 needs linearized readouts. If J_q varies strongly between cells, S4 is only locally valid. Report the linearization residual.
4. B's sparse-deviation model is exploratory. If the 126-item retest target is driven by a few dense factors, the expected and acceptable result is that compressed sensing adds nothing.

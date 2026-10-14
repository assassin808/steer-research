# Q2: Compressed sensing as a theory of multi-feature steering

Tags: [V] means I opened the arXiv or abstract page. [C] means a classic reference whose bibliographic details I checked by search but whose PDF I did not open. [N] means new as far as I could find (the derivations are elementary, but I found no prior statement of them for steering). [K] means known in CS, frame theory or communications, and only the application here is new.

## 1. Superposition as CS: what exists

- Elhage et al. 2022, *Toy Models of Superposition* (arXiv 2209.10652). Empirical study that names CS as the closest theory. It has no recovery theorems.
- Ivanitskiy, Jasper, King & Mixon, *Towards a mathematical theory of superposition* (arXiv 2608.27540) [V]. Features are a sparse binary x, stored as h=Wx and read out as ReLU(WᵀWx+b). For random supports, support recovery holds up to s ≍ d/log n. For worst-case supports there is an exact coherence threshold, attained by ETFs. This is the closest existing "superposition = CS" theorem.
- Hänni, Mendel, Vaintrob & Chan, *Mathematical Models of Computation in Superposition* (arXiv 2408.05451) [V]. All pairwise ANDs of m features can be computed with Õ(m^{2/3}) neurons. Width d can emulate circuits of width Õ(d^{1.5}). The companion "framework" posts by Vaintrob et al. are not on arXiv (unverified).
- Klindt et al., *From superposition to sparse codes* (arXiv 2503.01824) [V]. Proposes identifiability, then sparse coding, then interpretability as a pipeline.
- Worst-case vs random supports. Coherence-based ℓ1/OMP guarantees cap s at about 1/μ (Donoho & Elad, PNAS 2003 [C]). The Welch bound μ ≳ d^{-1/2} then gives the √d bottleneck. Random supports avoid it (Tropp, ACHA 2008, conditioning of random subdictionaries [C]).
- Union of subspaces and block sparsity, which is the right model for multi-dimensional features. Blumensath & Davies, IEEE-TIT 2009 [C], show that the number of measurements needed grows only logarithmically in the number of subspaces. Eldar, Kuppinger & Bölcskei (arXiv 0906.3173) [V] define block coherence.
- Manifold CS. Baraniuk & Wakin, FoCM 2009 [C]: random projections preserve a k-dim manifold with m ≍ k·log(N·V·R/τ), where τ is the reach (condition number).
- Dictionary identifiability. Spielman, Wang & Wright (arXiv 1206.5882) [V] cover complete dictionaries. Arora, Ge & Moitra (arXiv 1308.6273) [V] cover incoherent overcomplete ones. Together they justify SAEs in theory.
- Curved features. Engels et al. (arXiv 2405.14860) [V] find weekday and month circles. Wurgaft et al. (arXiv 2605.05115) [V] show that steering along the manifold stays natural while linear steering leaves it. Park, Choe & Veitch (arXiv 2311.03658) [V] argue the right metric is the "causal inner product".
- Steering side effects. van der Weij et al. (2403.05767) [V]: summing vectors into one mostly failed, while injecting at separate layers worked. Scalena et al. (2406.17563) [V] adapt the strength per property. MAT-Steer (2502.12446) [V] adds a soft orthogonality penalty. SAE-TS (Chalnev et al., 2411.02193) [V] reports that the **pseudo-inverse of a measured effect matrix underperforms** its heuristic (App. A.1). This is the empirical caution for §2.
- Closest edit-geometry analogue: LEACE (2306.03819) [V], a closed-form least-change edit for erasure.

## 2. Steering as an inverse problem

**Setup.** Take a residual stream h∈ℝᵈ and read-outs a(h)=Rh∈ℝᴺ with N≫d. The rows rᵢ are unit vectors: probes, SAE encoder rows, or diff-of-means directions whitened by M. Let G=RRᵀ and μ=max_{i≠j}|G_ij|. We want to move the target set T (|T|=s) by δ∈ℝˢ while leaving the other read-outs fixed. Norms use M (identity, or the causal inner product).

**P1 (minimum-norm edit = dual frame) [K].** If R_T has full row rank, then argmin{‖Δh‖_M : R_TΔh=δ} is Δh* = M⁻¹R_Tᵀ(R_TM⁻¹R_Tᵀ)⁻¹δ, the canonical dual frame. With M=I, the leakage onto j∉T is ℓ_j = G_{jT}G_{TT}⁻¹δ. Gershgorin gives |ℓ_j| ≤ √s·μ‖δ‖₂ / (1−(s−1)μ).

**P2 (when summing CAA vectors is optimal) [N, elementary].** Here CAA means Δh_CAA = R_Tᵀδ. Its target error is (G_TT−I)δ, with ‖·‖_∞ ≤ (s−1)μ‖δ‖_∞, and its leakage is G_{T^cT}δ. CAA equals P1 exactly when G_TT=I, i.e. the targets are mutually orthogonal in the M-metric. More generally, the Tikhonov dual Δh_λ = R_Tᵀ(G_TT+λI)⁻¹δ interpolates between the two: λ→0 gives P1 and λ→∞ gives λ⁻¹·CAA. This is the matched-filter / zero-forcing / MMSE trichotomy of multi-user precoding (Peel, Hochwald & Swindlehurst, IEEE-TCOM 2005 [C]). When the directions are estimated with noise of size σ, the optimal λ is about σ²·s. This may explain why the pseudo-inverse lost in SAE-TS: it amplifies noise when κ(G_TT) is large.

**P3 (protected set) [K/N].** Protect S⊇T with |S|≤d by requiring R_SΔh=(δ,0). Then Δh=R_S⁺(δ;0). Its energy ‖Δh‖² = δᵀ[(G_SS)⁻¹]_{TT}δ ≤ ‖δ‖²/(1−(|S|−1)μ) is a Schur-complement cost. If N>d, zero leakage on every feature is impossible: there are N−d unavoidable constraints. Protection pays only when coherence is *structured*, i.e. the heavy entries of G_{jT} sit on a few neighbours. Protecting the top-k neighbours by |G_jT| is then the greedy optimum.

**P4 (random-frame leakage, sharp) [N, standard proof].** Let the rᵢ be i.i.d. N(0,I/d). Since Δh* depends only on R_T, each r_jᵀΔh* for j∉T is exactly N(0,‖Δh*‖²/d). Concentration of the singular values of R_T gives ‖Δh*‖ ≤ ‖δ‖/(1−√(s/d)−t). Hence, with probability ≥ 1−η,
  max_{j∉T}|ℓ_j| ≤ ‖δ‖ (1+o(1)) √(2 log(2N/η)/d),
and the max over independent Gaussians matches this from below.

**P5 (curved features) [K pieces, N assembly].** Feature i is a kᵢ-dim manifold Mᵢ=φᵢ(Θᵢ) with reach τᵢ. Its read-out ψᵢ has Jacobian Aᵢ.
(a) Small moves: apply P1 with R_T replaced by the stacked Jacobians A_T. Then s becomes s_eff=Σ_{i∈T}kᵢ, and μ becomes the block coherence μ_B = max‖P_iP_j‖_op over the affine spans.
(b) Large moves along a geodesic of length L. The chord edit φᵢ(θ')−φᵢ(θ) stays in span(Mᵢ), so its leakage is ≤ μ_B·‖chord‖. A tangent (linear) step of length L instead lands roughly L²/(2τᵢ) off the manifold (Niyogi–Smale–Weinberger reach geometry [C]). That residual is the "unnatural" component reported by Wurgaft et al. It is curvature-controlled and goes to 0 as L/τᵢ→0.

## 3. Capacity conjecture

From P4, with s targets each of size τ (so ‖δ‖²=sτ²), relative leakage ≤ ε holds iff
  **s*(ε,d,N) ≈ ε²·d / (2 log N)**  (random-like dictionaries).
This matches the d/log n recovery threshold of Ivanitskiy et al.
- With JumpReLU/SAE thresholds bⱼ, only leakage above the threshold activates a feature. The condition becomes τ√(2s log N/d) ≤ b_min.
- For manifolds, replace s by Σkᵢ and log N by log N + k̄·log(diam/τ̄), following Baraniuk–Wakin covering numbers.
- The worst-case (adversarial T) analogue is s ≲ ε/μ, the √d regime.
- **Conjecture C1:** real LMs sit between the two regimes. Their capacity is set by the *tail* of |G_ij| (feature splitting and semantic clusters), not by μ.

**Measurement on models ≤4B** (e.g. a ~2B model with public SAEs, plus Qwen-1.5B/3B diff-of-means):
(i) Estimate G for SAE decoders, persona directions and diff-of-means directions, in both the Euclidean and the whitened metric.
(ii) Sample random T for s=1..64 and apply CAA, P1 and P1-Tikhonov.
(iii) Measure leakage at the layer (SAE activations), downstream (later-layer probes) and behaviourally (logit or judge).
(iv) Fit s*(ε) against ε²d/log N and against the predictions from the empirical Gram.

## 4. Person side: the encoding direction

Model: yᵢ = g(⟨aᵢ,θ⟩ + noise), where A (n×K) holds the item loadings. If K is about 5 and θ is dense, CS adds nothing, and Bayesian D-optimal adaptive design is right (n≥K).

CS becomes relevant in the gene-bank regime: K is large (many facets), and a person differs from the population prior in **s≪K** coordinates, θ = θ̄ + sparse.
- With a RIP design, n ≍ s·log(K/s) items suffice uniformly over all supports [K]. Binary or ordinal answers fall under 1-bit CS: Plan & Vershynin (arXiv 1202.1212) [V] need s·log(K/s) bits under heavy flips.
- D-optimality fails when n<K, because the Fisher information is singular. It optimizes for a *fixed* model and is not uniform over supports.
- Adaptivity improves minimax sparse estimation by at most log factors (Arias-Castro, Candès & Davenport, 1111.4646 [V]). It *does* help detection of weak signals (distilled sensing, Haupt, Castro & Nowak, 1001.5311 [V]).
- Recommendation: start with a random incoherent batch of about s·log(K/s) items for support discovery, then switch to D-optimal on the estimated support.
- I found no prior work that links RIP to MIRT item banks. This gap is unverified beyond one search.

The duality: reading (person) and writing (steering) are the same frame problem. The person side needs A to be RIP on sparse θ. The steering side needs Rᵀ to be well conditioned on sparse T.

## 5. Research questions

**RQ1 (leakage law).** Does relative leakage of P1, P1-Tikhonov and CAA follow ‖δ‖√(2 log N/d) (P4), or the empirical-Gram prediction G_{jT}G_TT⁻¹δ?
*Experiment:* 3 models × 3 dictionaries × s∈{1..64}. Report a predicted-vs-observed R² and s*(ε) curves, about 2 GPU-weeks on MIG.

**RQ2 (matched filter vs zero-forcing).** Is there a crossover noise level σ*, set by direction-estimation noise (bootstrap the diff-of-means over n prompts), where CAA beats the pseudo-inverse? Does Tikhonov dominate both, as P2 predicts?
*Experiment:* vary n∈{8..1024} prompts per concept and plot the success×fluency frontier.

**RQ3 (curved-feature capacity).** Steer the weekday circle, the month circle and helical numbers jointly with linear concepts. Test whether off-manifold error scales as L²/τ and whether capacity uses s_eff=Σkᵢ.
*Experiment:* fit manifolds (Wurgaft et al. protocol), estimate τ from local PCA, then compare chord, tangent and manifold edits.

## Paragraph for the Method section

> *Steering as constrained inversion.* Let R∈ℝ^{N×d} stack concept read-outs (rows normalised in metric M) with Gram G=RM⁻¹Rᵀ. To shift concepts T by δ while holding a protected set P fixed, we solve Δh = argmin‖Δh‖²_M + λ⁻¹‖R_TΔh−δ‖² s.t. R_PΔh=0, whose solution is a regularised dual frame $\Delta h=R_T^\top(G_{TT}+\lambda I)^{-1}\delta$ when P=∅, interpolating zero-forcing (λ→0) and scaled CAA (λ→∞). For curved concepts we replace rows by tangent Jacobians and use chord edits for large moves; the off-manifold error of linear steering is O(L²/τ). Under a random-frame model, worst-case leakage onto unsteered concepts is ‖δ‖√(2log N/d), predicting a simultaneous-steering capacity s*≈ε²d/(2log N), which we test against empirical Gram spectra of SAE, persona and difference-of-means dictionaries.


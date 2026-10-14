# Adversarial review of FLOW_FORMAL.md (v0) — 2026-10-14

Verdict: the skeleton holds up. Most first-order formulas are correct (velocity, spill-over K_ij/K_ii, least-interference). Five claims are wrong or ill-posed as written:
(a) G_lambda is not positive definite in general;
(b) "the path eps -> h + d*(eps)" and "the integral curve of V" are two different objects, and the draft treats them as one;
(c) using a probe gradient for grad T breaks the link to the readout and can make eta blow up;
(d) R^2_traits is neither bounded nor well-defined;
(e) the Taylor-radius proxy measures where the quadratic KL law breaks, which is a different thing from tilt-tracking reach.

Novelty: the single-trait velocity formula IS FishBack's Theorem 2 (they use G-pseudo-inverse, G+alpha*I and iterate), so Sec. 1 is not new. The most valuable thing found in this review is a new and provable fact: with a readout-consistent gradient, **eta <= Var_{p(.|h)}(s_tau)**, with equality iff the tilt is first-order reachable. This makes steerability a bounded, normalised quantity and turns "activation steering = projection of the tilt" into a theorem. I recommend making it the backbone of Paper C (Sec. 2 below).

---

## 1. Mathematical correctness

### 1.1 Objects (Sec. 0)

**(i) What is being perturbed is undefined.** "h ... patched in" mixes two things: patching a single position, and steering, which adds d at all positions. The second is a vector shared across positions, not one state. Replace

> State: h in R^d, residual stream at layer l (last token, or response-averaged).

with

> Intervention parameter: theta in R^d added to the residual stream at layer l at positions S (S = last prompt token, or all positions; state which). For a base input x, the readout is p_x(y | theta). Every object below (G, T, eta, K) is a function of (x, theta) and is computed at theta = 0 unless stated. "State h" is only shorthand for (x, theta).

This also makes G, grad T and the rest exactly computable by autograd. For the S = all-positions protocol, there is no single h at which a Jacobian could be evaluated.

**(ii) G: the index q is undefined, and so is the sequence-level case.**

> G(h) = sum_q J_q(h)^T (diag p_q - p_q p_q^T) J_q(h)

Replace with

> Next-token readout: G_x = J^T (diag p - p p^T) J, J = d logits / d theta (V x d).
> Battery readout (persons, Sec. 4): q ranges over a fixed set Q of probe prompts/questions, each with its own readout p_q; G = sum_{q in Q} J_q^T (diag p_q - p_q p_q^T) J_q, which is the Fisher information of the product distribution prod_q p_q.
> Sequence readout: G = E_{y ~ p(.|theta)}[ grad_theta log p(y|theta) grad_theta log p(y|theta)^T ] (Fisher of the continuation distribution; unbiased by sampling, or teacher-forced as sum over steps of per-step Fisher under model samples).

KL(p_theta || p_{theta+d}) = 1/2 d^T G d + O(|d|^3) holds for each of these (C^3 logits). The reversed KL agrees at second order and differs at third. That difference matters because the tilt (Sec. 2) uses KL(q || p) while the budget uses KL(p || p_d).

**(iii) Rank remark.** Since V >> d, G is generically full-rank in exact arithmetic. It is *numerically* low-rank, because diag p - pp^T weights tokens by p (FishBack reports an effective rank of 2-17% of d). The relevant risk is therefore ill-conditioning, not a structural kernel. Final-layer RMSNorm adds an exact null direction (radial) only when l is the last layer.

**(iv) G_lambda is not positive definite in general.**

> Positive definite for lambda > 0

False. G + lambda P_perp is PD iff G is PD restricted to range(P_par) (the local principal subspace). On-support directions to which the readout is insensitive, which is exactly the numerical low-rank case, stay (near-)singular. Replace with

> G_lambda = G + lambda P_perp + mu I, mu > 0 small (Tikhonov; PD for any lambda >= 0). Alternative: G + lambda Sigma_shrunk^{-1} with a shrinkage covariance estimate (PD by construction). P_perp depends on a rank k for the "local principal subspace"; k and the neighbourhood used to estimate Sigma(h) must be stated. Note that rho is defined but never used; either drop it or use it (e.g. lambda(h) increasing as rho(h) decreases).

Also state that P_perp and Sigma are Euclidean constructions. They break the reparametrisation invariance that makes G natural, and a quadratic penalty frozen at h does not penalise leaving support at finite distance (Open problem 4 is the consequence).

**(v) "A trait is a coordinate function ... level sets partition the states."** Every function's level sets partition its domain. For the "coordinate" and "foliation" reading you need regularity: grad T != 0 on the region considered, which makes the level sets C^1 hypersurfaces. Add: "on the open set where grad T != 0". At trait saturation (grad T -> 0) the flow is undefined (see 1.2).

**(vi) The probe surrogate breaks the theory.** See 1.3. Replace

> approximated by a differentiable probe trained to predict the judge score from h

with

> Its gradient is computed readout-consistently (no probe): next-token, T = sum_y p_y s_y, grad T = J^T (diag p - p p^T) s; sequence-level, grad T = E_{y~p}[ (s(y) - T) grad_theta log p(y|theta) ] (score-function identity, Monte Carlo with a baseline). A probe may be used only as a variance-reduced *estimate* of this quantity, and its gradient must be projected onto range(J^T) (equivalently, onto span of score functions).

### 1.2 Trait flow (Sec. 1)

**Velocity formula is correct.** max g^T d s.t. 1/2 d^T G d <= eps gives d* = sqrt(2 eps / eta) G^{-1} g, and with unit trait rate V = G^{-1} g / (g^T G^{-1} g). This is exactly FishBack Thm 2 (G^dagger q · rho / q^T G^dagger q) at lambda = 0.

**Two different objects are conflated.**

> The trait flow is the path eps -> h + d*(eps). Infinitesimally its velocity ... is V_tau(h)

In the "local form" (g and G frozen at h) the path eps -> h + d*(eps) is a straight ray in direction G^{-1}g(h). In the nonlinear form (exact T and KL), the argmax path can be non-unique or discontinuous, and it differs from the integral curve of V at second order (curvature of T and variation of G). Replace with

> Definition (trait flow). The trait flow is the solution of dh/ds = U_tau(h), U_tau = G_lambda^{-1} g / sqrt(g^T G_lambda^{-1} g) (unit speed in G_lambda), on the set where g != 0. Motivation: U_tau(h) is the direction of the budgeted problem's solution as eps -> 0. The budgeted path from a fixed start (argmax under the exact KL budget) is a different object: it agrees with the flow to first order in sqrt(eps) and is used only as the "one-shot" comparator.

Why unit-speed and not unit-trait-rate: V blows up as eta -> 0 (saturation), whereas U stays bounded. With U, s is the G_lambda arc length, approximately sqrt(2 * KL) for short paths, which gives the principled intensity coordinate that the adverb prompts failed to provide (DEVLOG PoC v2). Along U, dT/ds = sqrt(eta).

**The minimality claim is stated in the wrong metric.**

> it minimizes total output change for a given trait gain

True only for lambda = mu = 0. With lambda > 0 it minimises G_lambda-cost (output change + off-support penalty). Write "minimizes first-order G_lambda cost per unit trait gain; equals minimal output change (KL) only when lambda = mu = 0."

**Linear steering bullet.**

> Linear steering = assuming grad T and G_lambda constant (V constant).

Constant V is what linear steering assumes. A constant natural gradient is a sufficient condition for that, not the definition. CAA's mean-difference direction coincides with V only if Delta_mean is proportional to G^{-1} g, which is a testable claim (FishBack argues that CAA/ActAdd/ITI implicitly use approximate metrics). Replace with "Linear steering = a constant field V; CAA uses a constant direction that is natural-gradient only if mean-diff ∝ G^{-1}g." The IDEEA/CHaRS fields depend on the input x (context), not on h along the path. Say "V depends on x, piecewise constant / kernel-smoothed in x."

### 1.3 Steerability and the tilt (Sec. 2): the main new result

Take lambda = mu = 0, the pseudo-inverse, and readout-consistent g. Let u_d(y) = d^T grad_theta log p(y|theta) be the score function in direction d. Then d^T g = Cov_p(s, u_d) and d^T G d = Var_p(u_d). Therefore

    eta = g^T G^+ g = sup_d Cov_p(s, u_d)^2 / Var_p(u_d) <= Var_p(s)          (Cauchy-Schwarz)

Equality holds iff s - E_p s lies in span{u_d}, i.e. iff the tilt direction is in the tangent space of the steerable family. Three consequences follow.

1. **Bounded, normalised steerability.** eta_hat(x) = eta / Var_{p(.|x)}(s) lies in [0,1]. It is the squared cosine, in L^2(p), between the tilt tangent s - E s and the tangent space of {p(.|theta)}.
2. **"Activation steering realises a projection of the tilt path" becomes precise at first order.** The tilt q_t has dq/dt at t=0 equal to p(s - E s). The natural-gradient step G^+ g gives exactly the coefficients of the L^2(p)-orthogonal projection of s - E s onto span{u_d}. Beyond first order, "projection of a path" is ambiguous: the M-projection argmin_theta KL(q_t || p_theta) and the I-projection/variational argmax E_{p_theta}[s] - (1/t)KL(p_theta || p) give different curves at second order. Pick one (M-projection matches the reach definition).
3. **eta has two factors**, Var_p(s) ("how undecided the base model is about the trait") times eta_hat ("geometric reachability"). Paper C should test both factors separately (H1 below). A model that is confidently non-trait at x (Var ≈ 0) is unsteerable at first order regardless of the activation geometry. This is the clean mechanism behind "Euclidean geometry does not predict steerability".

With a probe gradient, g can have components outside range(J^T). These are amplified by 1/mu or 1/lambda, which inflates eta without bound. That is why the probe must be removed or projected (1.1 vi).

Replace the steerability bullet

> Steerability at h: eta_tau(h) = grad T^T G_lambda^{-1} grad T (squared trait gain per unit behavioral cost).

with

> Steerability: eta_tau(x) = g^T G^+ g with readout-consistent g. Under budget KL <= eps, the max first-order trait gain is sqrt(2 eps eta). Proposition: eta <= Var_{p(.|x)}(s_tau), equality iff the tilt is first-order reachable. Normalised steerability eta_hat = eta / Var_p(s) in [0,1]. With the support-aware metric, eta_lambda <= eta.

The tilt claim itself is correct and standard (Gibbs variational principle; MIS 2609.30218 states it, as the "ideal reward-tilted distribution", and explicitly notes that steering may not represent it). The tilt family is the exponential family through p, i.e. an e-geodesic.

**Reach.**

> r_tau(h) = sup{ t : KL(q_t || p(. | h + d*(t))) <= kappa }

Problems: d* is indexed by eps, not t (no mapping is given); the set need not be an interval; kappa has arbitrary units; t scales with the units of s. Replace with

> theta*(t) = argmin_theta KL(q_t || p_theta) (M-projection, computed by a few gradient steps warm-started along the flow). Tracking error e(t) = KL(q_t || p_{theta*(t)}) / KL(q_t || p_0) (relative; in [0,1] when theta = 0 is feasible). Reach r = inf{ t : e(t) > kappa }, with s standardised by its base std so that t is unit-free. First-order consequence: e(t) -> 1 - eta_hat as t -> 0. So eta_hat is the reach's t->0 limit, and r measures how long that holds.

**Taylor-radius proxy.** D(alpha) = KL(p_0 || p_{alpha v}) = a2 alpha^2 + a3 alpha^3 + ...; |a2/a3| is the radius where the *quadratic KL law* breaks (r_G, "metric validity radius"). It is not the tilt-tracking reach: tracking can stay good after the quadratic law fails, and it can fail inside r_G (when eta_hat is small). Rename it r_G and keep it as a separate feature. Practical notes: a3 has a sign and is asymmetric in ±alpha, so use min over ±. A3 is available exactly by nested forward-mode JVP (torch.func.jvp of jvp). Simpler and more robust: evaluate D at 4-6 alpha values and report the alpha where |D - a2 alpha^2| / D > delta, which costs only forward passes.

### 1.4 Several traits (Sec. 3)

- K_ij and the spill-over dT_j/dT_i = K_ij/K_ii along V_i: **correct** (first order).
- **New, cheap baseline:** with readout-consistent gradients, K_ij = <Pi(s_i - E s_i), Pi(s_j - E s_j)>_{L^2(p)}, with Pi the projection onto the score span. If both tilts are reachable, K_ij = Cov_{p(.|x)}(s_i, s_j), so the ideal spill-over is the regression slope of judge score j on judge score i across *base-model samples* at x. That needs no gradients. Hypothesis A-1 must beat this baseline as well as the Euclidean cosine; otherwise the metric adds nothing.
- Least-interference: **correct**. min d^T G d s.t. g_i^T d = 1, g_P^T d = 0 gives d ∝ G^{-1} g~_i with g~_i = g_i - g_P K_PP^{-1} K_Pi (residual in the G^{-1} inner product on covectors). Add: it requires K_PP invertible and g_i not in span(g_P), and the normalisation is V = G^{-1} g~_i / (g~_i^T G^{-1} g~_i). It holds the other traits fixed only to first order; along a finite path, re-project at every step.
- **Second-order: a structural fact the draft misses.** For fields with dT_k(V_i) = c_ki(h): dT_k([V_i, V_j]) = V_i(c_kj) - V_j(c_ki). Two consequences:
  (a) for plain flows, the trait-level order effect equals the *variation of the spill-over ratios* K_kj/K_jj along the other flow. This is computable and gives a predicted interaction to compare with the grid diff-in-diff;
  (b) for fully least-interference flows (c_ki = delta_ki), dT_k([V_i,V_j]) = 0. Order dependence is then invisible in trait coordinates at second order and shows up only in non-trait directions (G-length). So "path dependence" in Sec. 4.4 is zero by construction for the least-interference route. Say so, or define path dependence in G-length.
  Keep the data-based diff-in-diff as primary, as the draft says.

### 1.5 Between two samples (Sec. 4)

- **d(h_a, h_b):** well-defined as an infimum of G_lambda path length once G_lambda is PD (fix 1.1 iv). Useful facts: d_{G_lambda} >= d_G >= d_FR(p_a, p_b) = 2 arccos sum_y sqrt(p_a p_b), which is a closed-form lower bound. Computation does not need Jacobians: discretise the path h_0..h_K and minimise sum_k sqrt(2 KL(p(h_k) || p(h_{k+1}))) plus the support penalty over the waypoints. With next-token readout, two quite different persona states can have d ≈ 0, which makes R^2 0/0. For persons, use the battery readout of 1.1 ii.
- **R^2_traits is not bounded and not well-defined.**

> R^2_traits(a, b) = (Delta T^T K^{-1} Delta T) / d(h_a, h_b)^2

(1) K is evaluated at an unspecified point. (2) If G and g are frozen and T is linear, the straight segment delta is feasible for the constraint g^T d = Delta T, so the ratio is <= 1. With varying G and nonlinear T (any two genuinely different samples), the geodesic can be shorter than this first-order bound, so the ratio can exceed 1. (3) The draft's first-order validity requires h_a ≈ h_b, which contradicts the intended use (two people). Replace with the path-energy version, which is bounded by Pythagoras pointwise:

> Let gamma be the G_lambda-geodesic (or the discretised minimiser) from a to b. At each point, Pi(h) = G_lambda-orthogonal projector onto span{G_lambda^{-1} g_tau(h)}. Define
>     R^2_traits(a,b) = ∫ |Pi gamma'|^2_{G_lambda} ds / ∫ |gamma'|^2_{G_lambda} ds  in [0,1].
> Local version (h_a ≈ h_b, delta = h_b - h_a, G, K, g at the midpoint): R^2 = (g^T delta)^T K^{-1} (g^T delta) / (delta^T G_lambda delta), in [0,1]. Use the linearised g^T delta, not the observed Delta T, or the bound fails.

  The gap between the observed Delta T and the integral of g^T gamma' along gamma is zero (fundamental theorem of calculus). The gap between the observed Delta T and the local linearisation g^T delta is a diagnostic of nonlinearity worth reporting.
- **Interpolation (4.5):** flow-path points are not "between a and b" unless the path ends at b. Use geodesic points, or the least-interference trait route followed by a residual non-trait segment.
- **F-1 is ambiguous and at risk of circularity.** "Their trait vectors": T_tau is the *model's* trait functional, not the human's measured Big-Five score. Also, if Q (the readout battery) contains the questions whose answer differences are evaluated, R^2 and the target share data. Fix: Q for computing G and d is disjoint from the evaluation questions. State whether the trait vector is human-measured (Twin-2K-500 psychometrics) or T_tau(h_person), and test both.

---

## 2. Formal vs figurative: what is still undefined

Defined adequately once the fixes above are made: G, G_lambda (with mu), T and g (readout-consistent), U/V, eta, eta_hat, K, least-interference, r_G, d_{G_lambda}, R^2 (path version).
Still undefined or figurative:
1. The intervention protocol (positions S, layer l, last-token vs all positions). Every object depends on it.
2. The support: rho is never used; k, neighbourhood size and the estimator of Sigma(h) in d ~ 2048 are not given. "Region the model actually visits" is not yet an operational quantity. A minimal operational choice: kNN distance to a reference activation bank, normalised by the bank's own kNN distances.
3. The trait score s_tau: judge prompt, scale, and whether it is a function of the full continuation. Length, the sampling temperature for p(.|h), and the continuation length all change T, Var(s) and eta.
4. "Placing a person in the model" (which prompt, which position, which layer).
5. "Manifold" interpretation: nothing in the draft uses a manifold. Good. Keep it that way, or define the support set as the set where rho > threshold.
6. Finite-path guarantees: none. The flow's existence (g != 0) and whether it stays in support are open.
7. The relation between t (tilt temperature), eps (KL budget) and s (arc length): to first order s ≈ t sqrt(Var_p s) · sqrt(eta_hat) and eps ≈ s^2/2. State this.

---

## 3. Novelty (checked on arXiv pages, 2026-10-14)

| Claim in draft | Prior work | Status |
|---|---|---|
| Velocity G^{-1}g/(g^T G^{-1} g), pullback Fisher | **FishBack 2605.17231 Thm 2**: delta h* = rho G^dagger q / (q^T G^dagger q), G = J^T H J, q = J^T beta_W, Tikhonov G + alpha I, 30 iterated steps with frozen J ("iterative pullback steering"). Single concepts treated separately; no Gram matrix, no support term, no interference. | **Not new** (single trait). Cite as the origin of Sec. 1. The flow (integrated U with updated J) is a minor variant; FishBack reports re-computing J per step was worse, so this is an empirical risk. |
| Tilt optimum + Fisher quadratic | **MIS 2609.30218** (Entesari et al.): q* ∝ pi_ref exp(R/lambda) stated as the ideal; Fisher quadratic as a penalty (gradient step g - lambda F u, *not* natural gradient); single reward; per-prompt optimisation; no steerability prediction. | Not new as a statement. **The eta <= Var_p(s) bound, eta_hat and reach-as-tracking-error appear new** (not found). |
| Pullback metric on hidden states | **2507.19247** (Y. Zhang) Sec. 6: pullback g* = W^T(Diag p - pp^T)W, degenerate caveat; d_FR only between *output distributions*, bounded by a path integral of sqrt(g*) along the straight line. The draft's attribution "Fisher-Rao distance for LM hidden states" overstates it. | Change the citation wording: "pullback Fisher tensor and Lipschitz bounds; no hidden-state distance." |
| OT steering | AcT 2410.23054. arXiv only on the abs page (the draft's "ICLR 2025" was not confirmed there; verify the venue before citing). CHaRS 2603.02237 (ICML 2026): GMM + discrete OT + barycentric projection. | Correct as background. |
| Predicting steering effectiveness | **2602.01716** (Jafari, Xue, Salim): entropy-based Normalized Branching Factor + KL to concept vocabulary, LLM-judge ground truth. **2606.11599 "When is Your LLM Steerable?"** (Fan et al.): ASTEER, 1.4M steered generations, GBDT on before/after hidden-state features, ~0.7 macro-F1. | Both are key baselines for Paper C. Neither uses Fisher/tilt quantities. NBF is entropy-based and so close in spirit to Var_p(s): **H1 must show that eta_hat adds beyond Var_p(s) and NBF.** |
| Multi-trait Gram matrix | **2609.05800 Spillover-aware multi-value steering**: spill-over attributed to the Gram matrix of steering directions, with an inverse-Gram correction on weights (Euclidean, per the abstract; inner product not confirmed). MSRS 2508.10599; K-Steering (EMNLP Findings 2025, gradients of a multi-label classifier); A-LQR 2604.19018; AIMES 2609.30405. | **"Gram matrix predicts spill-over" is partly taken.** What remains new is the *metric* Gram K = g^T G^{-1} g (readout-consistent), its covariance limit, and the bracket formula for second order. A-1 must compare metric-K vs Euclidean-Gram (2609.05800) vs base-sample covariance. |
| Input-dependent gradient field | Steering Vector Fields 2602.01654 (gradient of a learned concept score, multi-attribute) | Euclidean-gradient field exists; the metric field is FishBack. |
| Trait-explained distance between samples / R^2_traits | Nothing found (persona vectors 2507.21509 and Big-Five activation papers use Euclidean projections). | **Appears new.** It is also the least validated part. |
| Off-support steering | "Steered LLM Activations are Non-Surjective" 2604.09839 | Cite for the support term. |

Net novelty after review: (1) eta <= Var_p(s) and eta_hat as normalised, gradient-only steerability, with reach as tracking error; (2) metric trait Gram + covariance limit + bracket second-order formula for composition; (3) bounded trait-explained share between samples. The flow definition itself is not a contribution.

---

## 4. Three hypotheses with minimal experiments (<= 2B, <= 2 GPU jobs, MIG OK)

Common setup: Qwen3-1.7B (fallback Qwen2.5-1.5B or Gemma-2-2B), layer l in {mid, late}, intervention = add theta at all positions after layer l. Judge scores come from a small local judge (or a lexicon-free classifier) on 64-token continuations at T=1. Traits: evil, sycophancy, plus one benign trait (formality) for checks.

**H1 (Paper C). Per-input steering success is predicted by Var_{p(.|x)}(s) and eta_hat, and Euclidean features add little.**
- For 300 prompts x 3 traits: sample 32 continuations per prompt, which gives Var_p(s) and the score-function g = mean((s - s̄) grad log p). G-vector products come from JVP/VJP on the same samples. Solve G^+ g with CG (20 iterations, mu small). That gives eta and eta_hat.
- Ground truth: steer with a fixed-KL-budget CAA *and* with the natural-gradient direction at 3 budgets. Measure the judge gain per sqrt(KL).
- Predictors: {Var_p(s), eta_hat, NBF (2602.01716), Euclidean: activation norm, cos(h, CAA dir), PCA distance}. Compare cross-validated R^2 / Spearman by nested regression.
- Falsifier: Euclidean features match eta_hat's predictive power, or eta_hat adds nothing beyond Var_p(s). Sanity check: the measured gain per sqrt(2 KL) at small budget should be <= sqrt(Var_p s) on every prompt (the bound), and close to sqrt(eta) for the natural-gradient direction.
- Cost: about 300 x 3 x (32 samples + ~40 JVP/VJP), a few GPU-hours on one MIG slice.

**H2 (Paper A). The metric Gram K_ij/K_ii predicts single-trait-steering spill-over better than Euclidean direction cosine/Gram, and better than base-sample score covariance.**
- Same prompts, pairs (evil, syco), (evil, formality), (syco, formality). Compute K per prompt from H1's quantities, plus the Euclidean cosine of CAA directions and Cov_p(s_i, s_j)/Var_p(s_i) from the 32 base samples.
- Steer trait i alone (natural-gradient and CAA, small budget). Measure Delta T_j / Delta T_i with the judge.
- Report per-prompt Spearman and the aggregate. Also run least-interference for i holding j: the measured |Delta T_j| should drop by the predicted factor.
- Falsifier: the covariance baseline matches K. In that case the activation metric is not needed for spill-over, which would be a finding in its own right.

**H3 (Paper A/B, the bounded R^2). The path-version R^2_traits between persona states predicts how much of their answer difference is captured by trait coordinates, on held-out questions.**
- 200 Twin-2K-500 persons. Persona prompt -> h at layer l. Battery readout Q = 30 questions (for G and d); evaluation questions = a disjoint 30.
- For 300 random pairs: discretised geodesic (K = 8 waypoints, minimise sum sqrt(2 KL)), then R^2 by the path formula with traits = the Big-Five-like T_tau (judge on battery continuations).
- Target: the fraction of variance in evaluation-question answer differences explained by regressing on Delta T (pairwise), vs explained by Euclidean distance and by human-measured Big-Five differences.
- Falsifier: R^2 is uncorrelated with the held-out explained share (rho < 0.1), or Euclidean Delta-projection does equally well.
- Run this only after H1 confirms that readout-consistent g is estimable with acceptable MC noise (check: split-half correlation of eta > 0.7).

---

## 5. Minimal edit list for v1
1. Sec. 0: intervention parameter theta; define q / battery / sequence G; G_lambda += mu I; regularity of T; readout-consistent g (drop the probe).
2. Sec. 1: define the flow as the ODE with unit-G-speed field U; separate it from the budgeted path; fix "minimizes output change" -> G_lambda cost; credit FishBack Thm 2.
3. Sec. 2: add the proposition eta <= Var_p(s), eta_hat; reach as relative M-projection tracking error with inf-exit; rename the Taylor radius r_G.
4. Sec. 3: covariance-limit baseline; the bracket formula; least-interference normalisation and invertibility.
5. Sec. 4: path-energy R^2 (bounded), local version with linearised Delta T; battery readout; disjoint Q for F-1; d_FR lower bound.
6. Sec. 5: correct the 2507.19247 description; add FishBack Thm 2, 2606.11599, 2609.05800, 2602.01654, 2604.09839; verify AcT's venue.

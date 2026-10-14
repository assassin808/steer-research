# Trait flows: formal definitions (v1, 2026-10-14; v0 reviewed in REVIEW_FLOW_FORMAL.md)

No manifold is assumed. Every object is computable by autograd or sampling. "Manifold"/"flow on a manifold" is interpretation only.

## 0. Objects

- **Intervention.** theta in R^d added to the residual stream at layer l at positions S (state which: last prompt token, or all positions). For a base input x the readout is p_x(y | theta). All objects are functions of (x, theta), evaluated at theta = 0 unless stated.
- **Readouts and behavioral metric G.**
  - next-token: G_x = J^T (diag p - p p^T) J, J = d logits / d theta;
  - battery (persons): G = sum_{q in Q} J_q^T (diag p_q - p_q p_q^T) J_q (Fisher of prod_q p_q);
  - sequence: G = E_{y ~ p}[ grad log p(y|theta) grad log p(y|theta)^T ].
  KL(p_theta || p_{theta+d}) = 1/2 d^T G d + O(|d|^3). G is generically full rank but numerically low rank (ill-conditioned).
- **Support-aware metric.** G_lambda = G + lambda P_perp + mu I (mu > 0 makes it positive definite). P_perp projects off the local principal subspace (rank k, neighbourhood stated) of a reference bank of natural activations; operational support distance = kNN distance to the bank, normalised by the bank's own kNN distances.
- **Trait.** T_tau(x, theta) = E_{y ~ p_x(.|theta)}[ s_tau(y) ], s_tau a fixed judge score (prompt, scale, continuation length and sampling temperature stated). A trait is a coordinate function, not a manifold.
- **Readout-consistent trait gradient** (score-function identity, no probe): g_tau = grad_theta T = E_p[(s_tau - T_tau) grad_theta log p] = Cov_p(s_tau, u), u = grad_theta log p.

## 1. Trait flow (definition)

dh/ds = U_tau,   U_tau = G_lambda^{-1} g_tau / sqrt(g_tau^T G_lambda^{-1} g_tau)   (unit speed in G_lambda), on {g != 0}.

- s is G_lambda arc length; for short paths s ~ sqrt(2 KL) (lambda = mu = 0), a principled intensity coordinate. Along U, dT/ds = sqrt(eta).
- U is the eps -> 0 direction of the budgeted problem max T s.t. 1/2 d^T G_lambda d <= eps; the finite budgeted path from a fixed start is a different object (agrees to first order), used only as a one-shot comparator.
- Minimises first-order G_lambda cost per unit trait gain; equals minimal output change only for lambda = mu = 0.
- The single-trait direction G^+ g / (g^T G^+ g) is FishBack (2605.17231) Thm 2; the flow definition itself is not a contribution. FishBack reports that recomputing J per step was worse: an empirical risk for integrating U.
- Linear steering = constant field. CAA's mean-difference direction is natural-gradient only if mean-diff ∝ G^{-1} g (testable). IDEEA/CHaRS: field depends on the input x, piecewise constant / kernel-smoothed in x.

## 2. Steerability and reach (Paper C)

**Proposition (steerability bound; lambda = mu = 0, pseudo-inverse).** With u_d = d^T u,

    eta := g^T G^+ g = sup_d Cov_p(s, u_d)^2 / Var_p(u_d) <= Var_{p(.|x)}(s),

with equality iff s - E_p s lies in span{u_d} (the tilt direction is first-order reachable). Proof: Cauchy-Schwarz in L^2(p).

- **Normalised steerability** eta_hat = eta / Var_p(s) in [0,1]: squared cosine in L^2(p) between the tilt tangent s - E s and the tangent space of the steerable family {p(.|theta)}.
- **Decomposition.** eta = Var_p(s) x eta_hat = (how uncertain the base model is about the trait at x) x (how much of that the intervention can reach). A clean mechanism for why Euclidean activation geometry alone cannot predict steerability.
- **Tilt and its projection.** The KL-regularized optimum is q_t ∝ p exp(t s) (stated in MIS, 2609.30218). dq_t/dt at 0 is p(s - E s); the natural-gradient step G^+ g gives exactly the L^2(p) projection of s - E s onto span{u_d}. Beyond first order use the M-projection theta*(t) = argmin KL(q_t || p_theta).
- **Reach** (tracking error). e(t) = KL(q_t || p_{theta*(t)}) / KL(q_t || p_0) in [0,1]; r = inf{ t : e(t) > kappa }, s standardised by its base std. As t -> 0, e(t) -> 1 - eta_hat: eta_hat is the reach's small-t limit, r measures how long it holds.
- **Metric validity radius r_G** (separate feature): radius where the quadratic KL law breaks; |a2/a3| from D(alpha) = a2 alpha^2 + a3 alpha^3 + ... (nested JVP), or the alpha where |D - a2 alpha^2| / D > delta from 4-6 forward passes (min over ± alpha).
- **Relations (first order).** eps ~ s^2 / 2; s ~ t sqrt(Var_p s) sqrt(eta_hat).

## 3. Several traits (Paper A)

- Metric trait Gram matrix K_ij = g_i^T G_lambda^{-1} g_j. First-order spill-over of flow i onto trait j: dT_j/dT_i = K_ij / K_ii.
- **Covariance limit (gradient-free baseline).** With readout-consistent gradients, K_ij = <Pi(s_i - E s_i), Pi(s_j - E s_j)>_{L^2(p)} (Pi = projection onto the score span); if both tilts are reachable, K_ij = Cov_{p(.|x)}(s_i, s_j), so ideal spill-over = regression slope of judge score j on i across base-model samples at x.
- **Least-interference flow** for i holding set P fixed: g~_i = g_i - g_P K_PP^{-1} K_Pi, V = G^{-1} g~_i / (g~_i^T G^{-1} g~_i); requires K_PP invertible and g_i not in span(g_P); first order only (re-project each step).
- **Second order.** For fields with dT_k(V_i) = c_ki(h): dT_k([V_i, V_j]) = V_i(c_kj) - V_j(c_ki). (a) For plain flows the trait-level order effect = variation of spill-over ratios along the other flow (a predicted interaction to compare with the grid diff-in-diff). (b) For fully least-interference flows it is zero in trait coordinates; order dependence then appears only in non-trait directions. Data-based diff-in-diff stays primary.

## 4. Between two samples

For two states a, b (two sentences; two people placed in the model with a battery readout):

1. Trait differences Delta T.
2. **Behavioral distance** d_{G_lambda}(a, b) = inf over paths of G_lambda length; d_{G_lambda} >= d_G >= d_FR(p_a, p_b) = 2 arccos sum_y sqrt(p_a p_b) (closed-form lower bound). Compute by discretising the path (K waypoints) and minimising sum_k sqrt(2 KL(p_k || p_{k+1})) + support penalty.
3. **Trait-explained share** (bounded): with gamma the geodesic and Pi(h) the G_lambda-orthogonal projector onto span{G_lambda^{-1} g_tau(h)},

       R^2_traits(a, b) = ∫ |Pi gamma'|^2 ds / ∫ |gamma'|^2 ds  in [0,1]   (norms in G_lambda).

   Local version (a ≈ b, delta = b - a, all at the midpoint): R^2 = (g^T delta)^T K^{-1} (g^T delta) / (delta^T G_lambda delta). The gap between observed Delta T and the linearisation g^T delta is a nonlinearity diagnostic.
4. **Path dependence**: measured in G-length (it is zero in trait coordinates for least-interference routes).
5. **Interpolation**: geodesic points, or a least-interference trait route followed by a residual non-trait segment (flow points are not "between a and b" unless the path ends at b).

## 5. Novelty after review

- Not new: single-trait natural-gradient direction (FishBack Thm 2); tilt optimum (MIS 2609.30218); pullback Fisher tensor on hidden states (2507.19247, which gives Lipschitz bounds but no hidden-state distance); Euclidean Gram-matrix spill-over correction (2609.05800); OT steering (AcT 2410.23054, venue to verify; CHaRS 2603.02237); input-dependent gradient fields (2602.01654); off-support steering (2604.09839).
- Appears new: (1) eta <= Var_p(s), eta_hat, and reach as tilt-tracking error; (2) metric trait Gram matrix with its covariance limit and the second-order bracket formula; (3) the bounded trait-explained share between samples.
- Paper C baselines: NBF (2602.01716), ASTEER (2606.11599). Paper A baselines: Euclidean Gram (2609.05800), base-sample covariance, MSRS, K-Steering.

## 6. Hypotheses (Qwen3-1.7B scale)

- H1 (C): Var_p(s) and eta_hat predict per-input steering success beyond Euclidean features and NBF; the bound eta <= Var_p(s) holds per prompt.
- H2 (A): metric K_ij/K_ii predicts joint-steering spill-over better than Euclidean Gram and base-sample covariance.
- H3 (A/B): path R^2_traits between persona states predicts held-out answer differences captured by trait coordinates (readout battery disjoint from evaluation questions; test human-measured and model T).

## 7. Still undefined

Intervention protocol (S, l); support estimator in d ~ 2048; judge details; how a person is placed in the model; finite-path existence and staying in support.

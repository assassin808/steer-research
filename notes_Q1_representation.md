# Q1: Which representation best explains a group of concepts?

Status: literature study and idea note. Every arXiv citation below was checked on its abstract page on 2026-10-14. Items marked [classical] are textbook results, not checked against arXiv. Items marked [spec] are my own conjectures and have not been proved or tested.

## 1. Map: five meanings of "best"

**(a) Explanatory / compressive**
- Voita & Titov, MDL probing (arXiv 2003.12298). Online and variational codelength measure "quality + effort". Accuracy alone fails to separate trained representations from random ones; MDL separates them.
- Xu et al., V-information (arXiv 2002.10689). Defines information that a fixed function class V can use, with PAC estimation. It is the natural way to score a *class* of representations.
- Tishby, Pereira & Bialek, Information Bottleneck (arXiv physics/0004057). Rate–distortion trade-off between compressing X and keeping the information about Y.

**(b) Causal / interventional**
- Geiger et al., causal abstraction foundations (arXiv 2301.04709). Shows that many interpretability methods are special cases of abstraction under interventions.
- Geiger et al., DAS (arXiv 2303.02536). Learns a rotated subspace whose interchange interventions implement a high-level variable.
- Makelov, Lange & Nanda (arXiv 2311.17030). Warning: a subspace patch can produce the target behavior through a dormant parallel pathway. Interventional success therefore does not prove that the representation is the one the model uses.
- Wurgaft et al. (arXiv 2605.05115). Steering along a fitted manifold keeps outputs close to natural model behavior. Linear steering leaves the manifold.
- Multi-attribute steering: K-Steering (Oozeer et al., arXiv 2505.24535), a gradient-based nonlinear classifier; MSRS (Jiang et al., arXiv 2508.10599), orthogonal per-attribute subspaces; Sequential Adaptive Steering for Big Five traits (Hoppe et al., arXiv 2603.03326). All three report interference from naive addition. None of them gives a criterion for *which* representation class to use.

**(c) Identifiable / unique**
- Locatello et al. (arXiv 1811.12359). Unsupervised disentanglement is impossible without inductive bias or supervision.
- Khemakhem et al., iVAE (arXiv 1907.04809). Conditioning on an auxiliary variable (for example a label) gives identifiability up to simple ambiguities.
- Squires et al. (arXiv 2211.16467). Linear mixing of linear latent SCMs: one intervention per latent suffices, and this is tight.
- von Kügelgen et al. (arXiv 2306.00542). Nonparametric case: two perfect interventions per variable suffice.
- Roeder, Metz & Kingma (arXiv 2007.00810). Discriminative models are identifiable up to a linear map.
- Marconato et al. (arXiv 2410.23501). A linear property of a next-token predictor holds in all distribution-equivalent models or in none of them.
- Rajendran et al. (arXiv 2402.09236). Concepts can be identified from diverse conditional data, which links causal representation learning to LLM concepts.

**(d) Geometric fidelity**
- Park, Choe & Veitch (arXiv 2311.03658). Linear representations become unified under a "causal inner product" defined through the unembedding.
- Park et al. (arXiv 2406.01506). Categorical concepts form polytopes, and hierarchical relations become orthogonality.
- Engels et al. (arXiv 2405.14860). Weekdays and months lie on circles, and these features are causally used.
- Kantamneni & Tegmark (arXiv 2502.00873). Numbers lie on a generalized helix.
- Modell et al. (arXiv 2505.18235). Features can be continuous manifolds, and cosine similarity can reflect a feature's intrinsic geometry.
- Csordás et al. (arXiv 2408.10920). Gated RNNs use magnitude-based ("onion"), non-direction codes.
- Pfau et al., GEOMANCER (arXiv 2006.12982). Product-manifold factors can be recovered if the metric is known and each factor has nontrivial holonomy.
- Higgins et al. (arXiv 1812.02230). Disentanglement defined through a product of symmetry groups.
- Wattenberg & Viégas (arXiv 2407.14662). Survey of relational composition beyond feature vectors.

**(e) Behavioral alignment**
- The causal inner product (arXiv 2311.03658) and manifold steering (arXiv 2605.05115) both tie geometry to outputs.
- I found no work that scores a multi-concept representation class by how well its metric matches the output-distribution metric (for example a Fisher pullback). See 3.1.

**Gap.** Each line of work optimizes one criterion inside one class. None of them compares several classes against several criteria for the *joint* representation of two or more concepts.

## 2. Unified problem statement

Two concepts take values $i\in[I]$ and $j\in[J]$. Let $h\in\mathbb{R}^d$ be the layer-$\ell$ residual activation. Write $\mu_{ij}=\mathbb{E}[h\mid i,j]$ and $p_{ij}=p(y\mid i,j)$.

**Candidate classes** (a nested chain on seen cells):
- $\mathcal R_{\rm add}$: $\mu_{ij}=m+U a_i+V b_j$.
- $\mathcal R_{\oplus\mathcal M}$: $a_i=f_1(\theta_i)$ and $b_j=f_2(\phi_j)$, where $f_k$ is a parametric chart (circle, helix, polytope).
- $\mathcal R_{\rm AMMI}(r)$: $\mathcal R_{\rm add}+\sum_{k\le r}\lambda_k u_{ik}v_{jk}w_k$.
- $\mathcal R_\otimes$: $m+T(\tilde a_i\otimes\tilde b_j)$.
- $\mathcal R_{\rm full}$: $IJ$ free vectors (joint-cell model).

**Honest caveat.** On a finite grid with free $a_i$, $\mathcal R_{\oplus\mathcal M}$ adds nothing over $\mathcal R_{\rm add}$; the two reproduce the seen cell means equally. Manifold structure shows up only through (i) a low-dimensional parametrization, tested on *held-out values*, and (ii) the *path* used for steering between cells. The ANOVA main-effect decomposition cannot tell linear structure from manifold structure.

**Criteria.** For a fitted $R\in\mathcal R$:
- $J_{\rm MDL}(R)$: prequential codelength of held-out $h$.
- $J_{\rm int}(R)=-\mathbb{E}\,\mathrm{KL}\big(p_{i'j}\,\|\,p(y\mid do(h\leftarrow s_R(h;i\!\to\!i')))\big)-\lambda\,\Delta_{\rm off}$, where $\Delta_{\rm off}$ is the change on concept 2. This is interchange-intervention accuracy scored in output space.
- $J_{\rm id}(R)$: whether $R$ is unique up to a known group $G_R$ (0/1, or a stability measure across seeds and prompt splits).
- $J_{\rm geo}(R)$: held-out-value reconstruction error.
- $J_{\rm beh}(R)$: rank correlation between $\|\mu_{ij}-\mu_{i'j'}\|_{G}$ and $\mathrm{JS}(p_{ij},p_{i'j'})$.

**Dominance.** $R$ dominates $R'$ if $J_k(R)\ge J_k(R')$ for all $k$, at equal or lower description length, with at least one inequality strict. Because the classes are nested, raw fit is monotone. All comparisons must therefore be held-out or penalized. The object of study is the Pareto frontier $\mathcal P\subset\mathcal R$ for each (model, layer, concept pair). The scientific question is *when the frontier collapses to a single class* (the criteria agree) and when it does not.

## 3. Proposed research questions

### 3.1 Compression and steering select different classes (Fisher-weighted interaction) [spec; the local result is near-elementary]

**Statement.** Let $G_{ij}=J^\top F J$ be the Fisher pullback metric of the output distribution at $\mu_{ij}$. To second order, additive steering $i\to i'$ at fixed $j$ satisfies

$$\mathrm{KL}\big(p_{i'j}\,\|\,p(\cdot\mid do(\mu_{ij}+\alpha_{i'}-\alpha_i))\big)=\tfrac12\,\Delta\gamma^\top G\,\Delta\gamma+O(\|\Delta\gamma\|^3),\qquad \Delta\gamma=\gamma_{i'j}-\gamma_{ij}.$$

MDL in the Euclidean metric selects $\mathcal R_{\rm AMMI}$ or $\mathcal R_{\rm full}$ whenever $\rho_E=\|\gamma\|^2/\|\mu-m\|^2$ is large. Steering needs interaction terms only when $\rho_G=\sum\gamma^\top G\gamma/\sum(\mu-m)^\top G(\mu-m)$ is large.

- **Conjecture.** The two criteria disagree exactly when $\gamma$ concentrates in low-Fisher (dormant) directions (then MDL says "entangled" but additive steering works) or in high-Fisher, small-norm directions (then MDL says "additive" but additive steering fails).
- **Why novel.** The causal inner product is a single global metric defined from the unembedding. Makelov et al. describe dormant directions only qualitatively. No existing work uses a Fisher-weighted interaction fraction to predict when multi-concept steering fails.
- **Test (≤4B, e.g. Qwen2.5-1.5B, Gemma-2-2B).** Estimate $G$ with low-rank JVP/VJP sketches at each $\mu_{ij}$. Across layers × concept pairs (weekday×count, optimism×sycophancy), check whether Spearman($\rho_G$, joint-minus-additive steering gap) > Spearman($\rho_E$, gap). If not, that is a negative result, and it is still informative.

### 3.2 Identifiability of direct-sum-of-manifold concept codes from single-concept interventions [spec; the proof sketch needs checking]

**Statement.** Suppose $h=m+\sum_{k=1}^K E_k(z_k)+\varepsilon$, where each $E_k:\mathcal M_k\to\mathbb{R}^d$ is an injective immersion. Let the affine hulls $S_k=\mathrm{span}\,E_k(\mathcal M_k)$ form a direct sum. Allow arbitrary dependence among the $z_k$ observationally. Suppose that for each $k$ we observe one soft intervention that changes only $p(z_k)$, and that this intervention is *generic*: the covariance change has column space equal to $S_k$.

Then $\{S_k\}$ is identified. The oblique projection $P_{S_k\parallel\oplus_{l\ne k}S_l}h$ recovers $E_k(z_k)$ up to translation, so each $\mathcal M_k$ is identified up to diffeomorphism.

- **Sketch.** Under intervention $k$, both $\Delta\mathbb{E}h$ and $\Delta\mathrm{Cov}\,h$ lie in $S_k$. The direct-sum condition makes the projections unique.
- **Counterexample.** If $\gamma\ne0$ (the spans overlap), the result fails. This gives a direct test of "product vs entangled".
- **Why novel.** It generalizes Squires et al. (1-D linear latents) to multidimensional manifold-valued factors under linear superposition. Unlike GEOMANCER, it needs no known metric and no holonomy. It applies when concept labels are latent (traits of real people), which is where the ANOVA's supervised identifiability no longer holds.
- **Test.** Use persona prompts that change exactly one trait. Compare the estimated $\hat S_k$ with the ANOVA main-effect subspaces (principal angles). Fit a circle/helix chart to $P_{S_k}h$. Verify on synthetic data first.

### 3.3 Model-selection test: additive vs manifold-sum vs AMMI vs full [method; mostly provable]

**(i) Bias fix.** The plug-in estimate $\hat\rho$ is biased *upward*. Noise variance enters in proportion to degrees of freedom, and the interaction term has the most: $(I-1)(J-1)$. Split the prompts into independent halves A and B and use
$$\hat\rho_\times=\langle\hat\gamma^A,\hat\gamma^B\rangle\big/\langle\hat\mu^A-\hat m^A,\hat\mu^B-\hat m^B\rangle,$$
which is unbiased in numerator and denominator under independent noise.

**(ii) Rank and class selection.** Choose the AMMI rank $r$ by held-out cell reconstruction (leave one cell out). Choose the manifold class by leave-one-*value*-out prediction with $f_k$ a Fourier or helix basis versus free $a_i$.

**(iii) Behavioral holonomy.** Steer around the loop $(i,j)\to(i',j)\to(i',j')\to(i,j')\to(i,j)$. In $\mathcal R_{\rm add}$ the loop closes. The residual equals the interaction contrast $\gamma_{ij}-\gamma_{i'j}-\gamma_{ij'}+\gamma_{i'j'}$. Measure it in output space (KL) as a test of non-additivity.

**Why novel.** Tukey's one-degree-of-freedom test and AMMI F-tests [classical] are for scalar responses. I found no prior cross-fitted, high-dimensional version paired with an output-space holonomy check.

## 4. Suggested minimal edits to the Method section

> **Representation selection.** Rather than fixing the representation a priori, we treat it as an object of inference. Let $\mathcal R=\{\mathcal R_{\rm add}\subset\mathcal R_{\oplus\mathcal M}\subset\mathcal R_{\rm AMMI}(r)\subset\mathcal R_{\rm full}\}$ denote nested models of the cell means $\mu_{ij}$, where $\mathcal R_{\oplus\mathcal M}$ constrains main effects to parametric charts $\alpha_i=f_1(\theta_i)$, $\beta_j=f_2(\phi_j)$. Each fitted $R$ is scored on held-out data by (i) codelength $J_{\rm MDL}$; (ii) interventional fidelity $J_{\rm int}=-\mathbb{E}\,\mathrm{KL}(p_{i'j}\,\|\,p(\cdot\mid do(s_R)))$ with an off-target penalty; (iii) geometric fidelity $J_{\rm geo}$ under leave-one-value-out; and (iv) behavioral alignment $J_{\rm beh}$ between $G$-distances of cell means and output divergences, where $G=J^\top FJ$ is the Fisher pullback metric. We report the Pareto set over $\mathcal R$ for each layer and concept pair. Interaction strength is estimated with the cross-fitted $\hat\rho_\times$ to remove the noise bias of plug-in estimates. We also report its Fisher-weighted analogue $\rho_G$, which we hypothesize predicts the gap between additive and joint-cell steering better than $\rho_E$.

Also in the Method section: (a) state the caveat that manifold structure cannot be identified from seen-cell ANOVA alone; (b) add the holonomy loop as a steering diagnostic; (c) for latent-trait settings (real people), add single-trait interventional prompts and the oblique-projection estimator of §3.2.

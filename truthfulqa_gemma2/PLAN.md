# TruthfulQA + Gemma2-2B: prediction track (H1)

Status: scaffold, not yet run. Script: `eta_truthfulqa.py` (untested; needs a GPU node).

## Question

Per question x, how much of the steering inefficiency is (a) the trait's own spread under the model's sampling distribution, (b) the intervention's reach at this layer, (c) the chosen direction? FLOW_FORMAL.md §2 gives the decomposition

    efficiency(v, x) = Var_p(s) * eta_hat(x) * cos_G^2(v, V(x)),   V = G^+ g,   eta_hat in [0,1].

This experiment tests whether a cross-fitted eta_hat(x) predicts realized steering gain per question, beyond Var_p(s) and cheap Euclidean features. It does not yet claim a leaderboard method.

## Setup

- Data: TruthfulQA generation split (817 questions), standard 6-shot QA prefix, same protocol as IDEEA (arXiv 2609.02089; venue not verified).
- Model: google/gemma-2-2b (d_model 2304, 26 layers). Layers: 12 (mid), 20 (late). Check IDEEA's exact checkpoint (base vs instruct) before comparing numbers.
- Intervention: theta in R^d added to the residual stream after layer l at all positions (prompt and generated).
- Trait score s(y) in {0,1}: truth AND info, from the two TruthfulQA judges (allenai/truthfulqa-truth-judge-llama2-7B, ...-info-judge-llama2-7B; repo names to verify). Same metric as IDEEA's headline (truth x info).
- Per question: n = 32 samples at T = 1, max 48 new tokens, stop at newline.

## Quantities per question

1. Var_p(s) = p(1-p) from the 32 judged samples.
2. Score features u_k = grad_theta log p(y_k | x, theta=0) in R^d, one backward pass per sample.
3. eta_hat_cf: split samples in two halves; ridge-regress s on u on half A; held-out R^2 on half B; swap and average; floor at 0. Plain in-sample eta_hat is identically 1 when n - 1 < d, so only the cross-fitted value is reported.
4. Direction V_A = ridge coefficient from half A (equals G_lambda^{-1} g with the sample covariance). Cost-normalised step theta = alpha V_A with alpha set so that the predicted KL = eps, eps in {0.01, 0.05, 0.2}.

## Ground truth

Fresh 32 samples under theta = alpha V_A. Realized gain = mean(s | theta) - mean(s | 0). Realized KL(p_theta || p_0) estimated on the same samples (two forward passes). Report gain per sqrt(2 KL).

## Predictors compared (nested regression, cross-validated Spearman / R^2)

- Var_p(s); eta_hat_cf; mean next-token entropy of the base model (NBF-style proxy, arXiv 2602.01716 is the real one); Euclidean: activation norm at the last prompt token, cosine to a global CAA direction fitted on a disjoint half of the questions.

## Sanity checks

- Realized gain per sqrt(2 KL) should not exceed sqrt(Var_p(s)) at small eps (the bound).
- Split-half correlation of eta_hat_cf across independent sample sets > 0.7 (otherwise the estimator is noise).
- Questions with Var_p(s) = 0 are reported separately: first-order unsteerable by construction.

## Falsifiers

- eta_hat_cf adds nothing over Var_p(s) in held-out prediction of realized gain.
- Euclidean features match eta_hat_cf.
- eta_hat_cf split-half reliability is low at n = 32 (then raise n or restrict to a low-dimensional sketch of u).

## Step 2 (only if the prediction track shows signal)

Method track: natural-gradient direction V(x) with a KL budget vs CAA, ITI, SEA, IDEEA at equal KL on truth x info. Dictator game (IDEEA's 4 social traits) for multi-trait interference K_ij.

## Cost estimate

817 questions x 32 samples x (1 generation + 1 backward) on a 2B model, prompt about 300 tokens: order of 1-2 GPU-hours on one A100-class card, plus judging. Steering evaluation adds 3 budgets x 32 samples per question per layer.

# Formula and source audit — 2026-11-05

## Source status

Original paper:
https://proceedings.neurips.cc/paper_files/paper/2025/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html

PDF:
https://papers.nips.cc/paper_files/paper/2025/file/97b73904e88cc1dc0a3485595eda3753-Paper-Conference.pdf

Official page also links a supplementary ZIP:
https://proceedings.neurips.cc/paper_files/paper/2025/file/97b73904e88cc1dc0a3485595eda3753-Supplemental-Conference.zip

The paper's reproducibility checklist answers Yes for code/data access. This
session found the official supplementary link, but web/working-container ZIP
retrieval failed; its contents could NOT be inspected. Exact-title web searches
and GitHub repository search returned no verifiable author-owned code repository.
This does NOT establish that the authors have not released code. No third-party
repository was substituted and no code was claimed to be copied from that ZIP.
The current implementation is independently written from the equations, with
explicit task adaptations below. Later official-source comparison remains open.

## Implemented mapping

| Original equation | Role here | Implementation |
|---|---|---|
| 7 | Hard-label Level-2 empirical risk; optional extra baseline | clean_weight (default 0) |
| 8 | Expectation of ordinary CE under Dirichlet | edl_expected_cross_entropy |
| 9 | VRM expectation | random permutation, beta/noise draws, batch mean |
| 10-11 | Strong real-sample Mixup | vicinal_pairs, matching task adaptation |
| 12-13 | Noise-augmented VRM | noise_pairs; two-view feature-space noise, uniform binary target |
| 14 | Dirichlet expected CE | sum t_k * (digamma(S)-digamma(alpha_k)) |
| 15 | Vicinal + noise | retained, plus existing FUME task loss |
| 51-59 | Local concentration gradients | formula-51 regression test; NOT Adam monotonicity |
| 62 | Mean alpha/S | mean_probability |
| 63 | Entropy of mean | total_entropy |
| 64 | Expected categorical entropy | aleatoric_entropy |
| 65-66 | Mutual information = TU - AU | epistemic_mi |
| 67-68 | Dirichlet differential entropy | dirichlet_entropy |
| 69-71 | Subjective-logic quantities; K/S | vacuity (K=2 in this task) |
| 72 | KL regularizer for other EDL baselines | deliberately not added to default VLS |

Equations 1-6 are definitions, not six missing loss branches. This release does
not silently add per-label VLS; it preserves the binary image-text relevance
head. Class targets are [match, nonmatch], never a normalized multi-hot class
vector. Default clean=0 does not disable original FUME supervision.

## Mathematical checks vs unproved task claims

The adopted source expected CE is unchanged. The following are independent audit
notes, NOT changes to the paper and NOT claims that its experiments are invalid.

1. For independent zero-mean endpoint perturbations with equal covariance, and
   lambda ~ Beta(beta,beta) independent of them,
   E[lambda^2+(1-lambda)^2]=(beta+1)/(2beta+1)=11/21 for beta=10.
   This is a noise covariance identity, not a test-risk or mAP improvement factor.
   The paper's Eq.47-to-48 substitution appears arithmetically inconsistent.
2. The expected CE in Eq.14 is linear in the target vector with the prediction
   fixed. Its target Hessian is zero. Therefore the paper's target-Taylor route
   in Eq.16-18 is not used here to claim a positive risk floor for this loss.
3. Cross-modal pair interpolation need not yield its true match probability.
   If t_mix - p*(x_mix) = b_mix + lambda*eps_a + (1-lambda)*eps_b,
   the conditional MSE includes b_mix^2, the two scaled variances, AND
   2*lambda*(1-lambda)*Cov(eps_a,eps_b). Shared image/text endpoints need not
   have independent errors. These terms are not assumed zero by the code.
4. Accurate evaluation of Dirichlet moments does not identify true data/model
   uncertainty. The added metrics assess prediction quality and ranking of
   errors/OOD; true source accuracy still needs a controlled or repeated-label
   protocol. No hard training-risk bound, rollback, or confidence filter is added.

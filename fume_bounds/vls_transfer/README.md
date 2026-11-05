# Complete VLS transfer for FUME (research prototype)

This is a **full initial implementation**, not an ablation-only partial component. It combines the two original components of Li, Chen & Yue (NeurIPS 2025): **strong vicinal supervision** and **Gaussian noise-augmented vicinal supervision**, both trained with their expected-CE Dirichlet loss (paper Eqs. 10–15). FUME's original fuzzy credibility and contrastive loss are retained, and *both image/text encoders receive VLS gradients*. The new evidence head is black-box and predicts binary **image–text relevance**, not a mutually exclusive category from INRIA's multilabel vectors.

## Important attribution and limits

- VLS's mathematical idea and hyperparameters are **from the paper**, not a new contribution. Cross-modal pair construction and dual-tower mixing are a **task adaptation**, not guaranteed faithful to the original supervised image classification input distribution. Mixup's interpolated relevance target need not equal real pairwise match probability.
- Official paper defaults confirmed: strong `Beta(10,10)` mixing; noise mixing `Beta(1,1)`; positive/nonmatching Dirichlet concentration `1+softplus(logits)`. Paper's full objective is `L_vicinal + L_noise`. FUME's own loss is kept as a separate original-task regularizer, `L_fume + vls_weight*(L_vicinal+L_noise)`.
- `noise_std=1.0` is an *implementation choice to be tuned for pre-extracted features*, not asserted to be the uniquely correct paper value.
- Every aligned sample participates in original FUME supervision and is also in the positive pair bank. Negatives are drawn from observed non-overlapping labels; no samples are filtered by confidence. With no negatives in a batch, VLS degenerates to positive-only supervision, which the logs expose.
- Zero additional semantic probing / Beta distillation / lower-bound solver / parameter rollback has been inserted. The VLS branch can be disabled term-by-term later for ablations without changing code.
- No real INRIA/CIFAR result is claimed. Local CPU tests check math and gradient flow. Unfiltered mAP is computed using embeddings actually updated by this training, not just an auxiliary head.

## Git integration

Place `vls_transfer.py` and `train_fume_vls.py` in `fume_bounds/vls_transfer/` (or run from this directory) and use an existing upstream installation:

```bash
cd fume_bounds/vls_transfer
python -m pytest -q
python train_fume_vls.py \
  --fume-dir /path/to/FUME/2025-CVPR-FUME \
  --dataset INRIA \
  --checkpoint /path/to/prior/encoder.pt \
  --epochs 20 --batch-size 128 --lr 5e-5 \
  --output /path/to/new_output_run
```

If no prior `encoder.pt` is available, omit `--checkpoint` to train FUME and the complete VLS branch together from scratch. A prior encoder checkpoint must contain both `state_dict` and `fixed_W`; the original FUME maps `W` as an ordinary non-parameter tensor, so accepting a checkpoint without `W` silently changes its semantics.

For reproducibility, new output folders are required. Per-epoch loss parts are written in `history.jsonl`, and model/head/config are saved to `vls_final.pt` and `config.json`. Test mAP is computed **only once at the end**, without threshold filtering or test-set checkpoint selection. `--no-eval` disables this final test pass.

## Controls available *after* the initial full run

```bash
# Vicinal-only ablation
python train_fume_vls.py ... --noise-weight 0

# Noise-only ablation
python train_fume_vls.py ... --vicinal-weight 0

# Add an optional ordinary clean pair EDL loss
python train_fume_vls.py ... --clean-weight 1

# Pair sampling, amplitudes, and source loss weight
python train_fume_vls.py ... --negatives-per-anchor 2 \
  --mix-beta 10 --noise-beta-clean 1 --noise-beta-random 1 \
  --noise-std 0.3 --vls-weight 0.5
```

`--noise-weight=0 --vicinal-weight=0` requires a nonzero clean term. Setting all terms zero is rejected. Use `--vls-weight=0` if testing untouched FUME training with this entrypoint (evidence head then remains unused).

## Key equations

Given a pair `(xI_i, xT_j)` with relevance `r_ij = 1[(y_i & y_j) != 0]`, virtual samples mix **both modalities with the same** lambda:

\[
\tilde{x}^{v}=\lambda x^{v}_{ij}+(1-\lambda)x^{v}_{kl},\quad
\tilde r=\lambda r_{ij}+(1-\lambda)r_{kl},\quad
\lambda\sim\mathrm{Beta}(10,10).
\]

Noise-augmented counterpart:

\[
\tilde{x}^{v}=\lambda x^{v}_{ij}+(1-\lambda)\epsilon^{v},\quad
\tilde r=\lambda r_{ij}+(1-\lambda)/2,\quad
\lambda\sim\mathrm{Beta}(1,1).
\]

`x^I` and `x^T` are **separate feature spaces** and are never added together. The evidence model sees their embeddings **after** the original FUME encoders.

\[
L_\mathrm{EDL}(\tilde r,\boldsymbol\alpha)=
\sum_{k=0}^1\tilde y_k[\psi(\alpha_0+\alpha_1)-\psi(\alpha_k)],\quad
\boldsymbol\alpha=1+\mathrm{softplus}(a).
\]

Note that all pairwise labels are computed only from training labels. Unknown/missing multi-label positives may appear as negatives; this is a limitation to evaluate, not a noise-modeling contribution.

## Evaluation plan

First run the whole configuration, then compare: original FUME, FUME+binary evidence with clean labels, VLS-only, noise-only, both VLS losses, optional clean term, matched capacity and training budgets. Compare full I→T/T→I mAP, match probability Brier/NLL, confidence/coverage at fixed recall, OOD and controlled known-ambiguity data. Do not use filtered mAP as the only trust metric. Real INRIA has no repeated ambiguity annotations, so its accuracy alone cannot prove correct epistemic/aleatoric attribution.

Original source: [Li et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html). Upstream: [FUME](https://github.com/siyuancncd/FUME).
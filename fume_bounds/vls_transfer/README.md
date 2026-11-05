# FUME + VLS: binary matching transfer and uncertainty evaluation

## Current scope (2026-11-05, v2)

This is an independent, task-adapted implementation of Li, Chen & Yue,
**Vicinal Label Supervision for Reliable Aleatoric and Epistemic Uncertainty
Estimation**, NeurIPS 2025. It is not the authors' official code or a complete
CIFAR reproduction. The existing binary matching task, two input-mixing branches
and original FUME loss are retained. No per-label VLS, new risk solver, parameter
rollback, Beta distillation, or novel uncertainty theorem is silently added.

Official paper and source availability audit: see [FORMULA_MAP.md](FORMULA_MAP.md).

## Train with the existing FUME checkout

```bash
git pull --ff-only origin main
cd fume_bounds/vls_transfer
pip install -r requirements.txt
python -m pytest -q
python train_fume_vls.py \
  --fume-dir /path/to/FUME/2025-CVPR-FUME \
  --dataset INRIA --checkpoint /path/to/encoder.pt \
  --epochs 20 --batch-size 128 --lr 5e-5 \
  --uq-pairs 20000 --uq-chunk 1024 \
  --output /path/to/new_run
```

Omit `--checkpoint` to train from scratch. With `--checkpoint`, training loads
FUME encoder weights and the ordinary Tensor `W`, initializes a NEW evidence
head and a NEW Adam optimizer. This is an encoder warm start, NOT exact optimizer
resume. `encoder.pt` must contain `state_dict` (or `fume_state`) and `fixed_W`.
Do not pass an encoder-only checkpoint to the standalone uncertainty evaluator.
The original upstream data paths and formats remain the user's responsibility.

Every original training row participates. Test data are evaluated once after
fixed-epoch training, not used to choose a checkpoint. Use `--no-eval` to skip
this final evaluation. Results require a new output directory.

## Evaluate an already trained VLS checkpoint without retraining

```bash
python evaluate_vls.py \
  --fume-dir /path/to/FUME/2025-CVPR-FUME --dataset INRIA \
  --checkpoint /path/to/run/vls_final.pt \
  --uq-pairs 20000 --uq-seed 20261105 --uq-chunk 1024 \
  --output /path/to/new_evaluation
```

Both v1 and v2 `vls_final.pt` files can be evaluated: the evaluator restores the
FUME encoder, fixed W and trained evidence head, inferring the head hidden size
from its state. v1 has no trained-head flag; do not evaluate a v1 baseline made
with `--vls-weight 0` as a trained evidential model. v2 explicitly rejects such
an untrained-head checkpoint. Evaluation does not modify checkpoint parameters.

## What is computed

Retrieval always uses ALL test queries and ALL test candidates, including paired
entries. Ranking uses cosine similarity of the trained FUME embeddings. AP is
zero when a query has no observed relevant candidate; it is not skipped. No
uncertainty threshold changes retrieval candidates or the mAP denominator.

The binary evidence head is evaluated separately. By default 20,000 ordered
pairs are uniformly drawn without replacement from the full test Cartesian pool,
using a fixed independent NumPy seed, with no class balancing or uncertainty
filtering. This keeps natural pair prevalence and includes cross-row positives
when drawn. Pair IDs and their hash are saved. `--uq-pairs 0` evaluates all pairs,
which can be expensive in memory, runtime, and exported files. Sampling for UQ
never reduces the set used by retrieval mAP.

Outputs:

- `vls_final.pt`, `history.jsonl`, `config.json`, `provenance.json`: trained model,
  configuration, consumed sample counts, source/checkpoint hashes.
- `test_metrics.json`: full unfiltered I-to-T and T-to-I cosine mAP.
- `pair_predictions.npz`: pair indices, hard relevance, alpha, mean probability,
  concentration, total entropy, aleatoric entropy, epistemic MI, Dirichlet
  differential entropy, and vacuity. Class order is **match, nonmatch**.
- `uncertainty_report.json`: binary Brier, two-class-summed Brier (2x binary
  Brier), predictive NLL, accuracy, probability ECE, relevance AUROC/AP, and
  per-uncertainty error-detection AUROC/AP/FPR95 and AURC/E-AURC.
- `risk_coverage_*.csv`: pairwise classification risk-coverage curves. These are
  NOT filtered retrieval mAP curves. All pairs contribute to exact AURC; the CSV
  is a maximum 1,000-point display grid.
- `evaluation_summary.json`: combined protocol and explicit OOD status.

All entropies use natural logs (nats). AU = expected categorical entropy;
EU_MI = total entropy - AU. Dirichlet differential entropy can be negative and
is a separate score, NOT added to AU. Vacuity = K/S; it is not substituted for
EU_MI. These are model-implied measures, not ground-truth AU/EU estimates.
Exact uncertainty ties use expected random ordering within a tie group, so
pair enumeration cannot artificially improve AURC. ECE bins positive match
probability, not top-label confidence. Undefined detection metrics with only one
class are JSON null. No confidence intervals assume that all pairs are iid.

## Optional genuine OOD data

Provide `--ood-pairs /path/to/ood_pairs.npz` to either entrypoint. This file must
contain `image_features [N, D_I]` and `text_features [N, D_T]`, with each row an
actual OOD pair, in the same feature/preprocessing spaces as training. The user
must justify the OOD split; a negative pair or a noisy training pair is NOT
silently labeled OOD. ID/OOD pair-selection protocols should be comparable.

The code saves `ood_predictions.npz` and `ood_report.json`, with OOD positive
for AUROC/AP/FPR95. Without an OOD file, status is `not_run_no_ood_data`; no OOD
result is invented. An OOD AUROC does not prove correct source disentanglement.

## Preserved training recipe and ablations

The default remains:

`L_total = L_FUME + L_vicinal + L_noise`.

Strong Mixup uses Beta(10,10); Gaussian-noise mixing uses Beta(1,1). The same
lambda mixes both views within their OWN pre-extracted feature spaces, before
the encoders. The evidence loss is the Dirichlet expected CE, alpha=1+softplus.
`clean_weight=0` follows paper Eq.15, which has no separate clean EDL term;
original FUME hard-label supervision is still present. Noise std=1 is a transfer
implementation choice, not a universal feature-scale recommendation.

```bash
# Change these flags on the training command; keep data/seed/budget matched.
--noise-weight 0                  # vicinal only
--vicinal-weight 0                # noise only
--clean-weight 1                  # extra clean EDL, not required by original VLS
--vicinal-weight 0 --noise-weight 0 --clean-weight 1  # hard EDL baseline
--vls-weight 0                    # original FUME path; skip random head's UQ
```

All term weights, mixing distributions, negative count and noise scale remain
configurable. Training negatives have no observed label overlap; missing labels
can create false negatives. Empty or non-binary training labels now raise an
error instead of silently declaring an invalid aligned pair positive.

## What has and has not been validated

See [TEST_REPORT.md](TEST_REPORT.md). Tests cover analytical values, sampling
checks, two-encoder gradients, tie handling, save/reload, the command-line
training-to-evaluation path, and a synthetic OOD-interface test. They are NOT
real INRIA results, GPU validation, or evidence that estimated AU/EU correspond
to their true sources. No default mixing, noise or loss setting is claimed to
improve real retrieval before that experiment is run.

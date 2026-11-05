# VLS v2 verification — 2026-11-05

## Executed in this session

Environment: Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5, pytest 9.0.2.
CUDA was unavailable. Parent revision: 4577e2e9ae625cdf9a8af9e656cc0cdbeb944808.

```text
OMP_NUM_THREADS=2 python -m pytest -q --junitxml=/mnt/data/vls_v2_junit.xml
43 passed in 19.08s
```

This count covers the 14 retained VLS transfer tests and 29 added cases (including
parameterized cases). It does not claim to rerun the separate historical v3/v4
controller suite, which is not modified by this change.

## Numerical and behavioral checks

- Beta(1,1): TU=log(2), AU=0.5, EU=log(2)-0.5, differential entropy=0, vacuity=1.
- Analytical Dirichlet differential entropy matches torch.distributions for
  K=2,3,20; AU matches 40,000 Dirichlet samples within the specified tolerance.
- Equal predictive means with different concentration have identical TU but
  distinct AU/EU/vacuity. Class permutations preserve scalar uncertainty.
- Finite statistics for concentrations up to 1e6 and logits +/-1000; malformed
  concentrations, NaN/Inf configurations and invalid supervision are rejected.
- Expected-CE gradients match the trigamma formula; Beta(10,10) perturbation
  variance coefficient is independently checked as 11/21. Neither check claims
  a test-risk or mAP bound.
- AUROC agrees with brute-force pair comparisons, including ties. Constant
  uncertainty scores give tie-order-independent AURC. One-class detection
  metrics are null, not invented zeros or NaNs.
- Full unfiltered cosine mAP, zero-relevance-query handling, deterministic uniform
  UQ pair selection, strict JSON serialization, and prediction/curve export.
- Actual PyTorch Adam updates reach both mock FUME towers and the evidence head.
- CLI integration: train a small mock FUME repository, save all model components,
  independently reload with evaluate_vls.py, and compare all exported predictions
  within 1e-6 and mAP exactly. The fixed W is restored with the encoder.
- Synthetic paired-feature NPZ exercises the optional OOD interface. This is an
  INTERFACE test, not a genuine OOD benchmark or a performance claim.
- Evaluation preserves model parameters and training/eval mode. Existing result
  files are not overwritten. Encoder-only checkpoints are refused for evidence
  evaluation. Disabling VLS skips the random untrained head's uncertainty report.

## Not executed / not claimed

No real INRIA/FUME dataset training, GPU run, CIFAR reproduction, real OOD
experiment, AU/EU ground-truth identification, or retrieval improvement was
performed. End-to-end CLI tests use deliberately small mock upstream modules,
not the full official FUME network or the user's server checkpoint.

The official VLS supplemental ZIP link was located, but its bytes could not be
retrieved in this environment; no verified author GitHub repository was found.
This is independent equation-based task adaptation, not official-source parity.

Original VLS training defaults remain unchanged. This change primarily closes
the missing uncertainty-evaluation path and adds input/checkpoint safeguards.

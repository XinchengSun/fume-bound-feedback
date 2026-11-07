# Evidence A/B implementation verification — 2026-11-07

Parent: `e3a45df9ee075a015a6202d928948870ba29a4ae`.
Environment: Python 3.13.5, PyTorch 2.10.0+cpu; CUDA unavailable.

## Executed

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q tests/test_evidence_ab.py
25 passed in 40.30s
```

This is the new suite, not a claim to rerun every historical v3/v4/v5 suite.

Covered:
- Matching has the same initial encoder/W hash as category, but no category evidence layers; category has no pair head.
- Pair targets equal complete multi-hot label intersection; semantic nonaligned positives supported; overlapping labels never become negative.
- No-negative and singleton batches do not manufacture negatives or remove original records.
- Softplus+1 prior, final Beta mean, and pair inference without labels.
- Category+posterior loss and every gradient match the unchanged v5 experiment objective exactly.
- Under direction-BCE control, clean classification/contrastive values and backbone gradients match across arms before evidence training.
- Joint vs staged backward for both arms; irregular pair chunks equal the full-pool mean; both encoders and active evidence parameters receive real Adam updates; W remains fixed.
- Mixup alone and noise alone each send gradients through both matching encoders and the pair head.
- Undefined category-only options for matching are rejected rather than silently ignored.
- All-query retrieval and fixed uniform pair UQ export; separate native target reports, TU=AU+EU, no false category posterior in matching.
- Unsupervised random pair-head UQ reporting disabled when no evidence objective is active.
- CLI train -> inference checkpoint reload -> evaluation, for both arms.
- One epoch + resume to two equals continuous two-epoch model states and DataLoader RNG, for both arms; 13 rows with batch 5 all participate.
- Legacy checkpoints, resume overrides and output overwrites rejected.
- The actual compare_evidence.sh executes both tiny arms, preserves equal backbone initialization, uses declared common direction-BCE, and creates a same-epoch validation summary.
- Baseline model.py/losses.py/experiment.py/metrics.py/data.py bytes verified against the parent Git blob SHAs; these source files are not modified in the commit.

## Full-size synthetic smoke

`python smoke_matching.py` ran one real CPU Adam step with default image4096/text1000/hidden4096/4-hidden-layer/K100 dimensions, batch2 synthetic data, complete class BCE, original CL, matching Mixup, matching noise and an active wrong-evidence KL.

- Trainable parameters: 122,440,194.
- Loss: 2.560885170164307.
- Two tower output-layer gradient norms: 0.5615471005 and 0.6807247400.
- Pair output-layer gradient norm: 0.0476183407.
- All three monitored parameters changed by approximately 5e-5 after Adam.
- No category evidence layers exist in this model.
- Exact record: matching_full_smoke.json. Its local one-step time is NOT a prediction of server runtime.

## Not executed / not claimed

No real INRIA data, pretrained server checkpoint, GPU training, real OOD benchmark, hyperparameter search, AU/EU ground-truth identification, or retrieval improvement was measured. Tests use synthetic data and complete-sized or deliberately small networks. Pair soft-label interpolation remains a modeling assumption. Same epoch/data budget is not identical compute; native pair and semantic-label uncertainty scores have different targets.

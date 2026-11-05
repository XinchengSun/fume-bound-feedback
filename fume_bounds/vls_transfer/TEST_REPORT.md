# VLS complete transfer — local verification (2026-11-05)

- `python -m pytest -q`: **14 passed**, no failures.
- Covers both paper VRM terms, binary Dirichlet expected cross-entropy, common-λ mixing across modalities, noise soft labels, multihot relevance, every original row in the positive-pair pool, matching-pair negative selection, both encoder gradient paths, actual Adam update, term switches, exact unfiltered AP@all definition, and an end-to-end CLI smoke against a temporary mock FUME repository.
- CLI smoke trained all 24 training rows, used both VLS terms, wrote full checkpoint and full-query two-way test mAP without filtering.
- Python 3.13.5, torch 2.10 CPU; **no real FUME upstream benchmark run, GPU run, INRIA training, calibration validation, or evidence of mAP improvement**.
- Limitations: strong pair-level Mixup target is a task adaptation; matching probability under mixed cross-modal inputs need not equal interpolated label; the original VLS theorem is not automatically transferred to this dataset and sampling protocol.
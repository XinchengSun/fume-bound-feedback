# Full-Unity experiment — implementation verification boundary

The experiment encodes the exact user-approved decomposition:
L_total = 1*L_CL + 1*L_Cls + (1*L_E + 1*L_Mix + 1*L_Noise), KL=0.

## Static checks completed before GitHub commit
- core.objective continues to call historical E minimal_objective; it overrides only the clean CL coefficient through explicit cl_weight, with backward-compatible default .05.
- run.py exposes --cl-weight, storing the weight in provenance and forwarding it into objective.
- The dedicated launcher explicitly sets --cl-weight 1, --pair-clean-weight 1, --vicinal 1, --noise 1, --error-kl 0, and preserves original matching model/pair sampling and E membership.
- The read-only verifier checks per-epoch five-loss sum, all weights, 150 epochs, 4500 updates, source/sample records, gradients, and full 4366-query test protocol.
- No historical checkpoint/experiment archive is edited.

## Actual execution status
- **No remote GPU1 execution occurred from this ChatGPT session.**
- **No real INRIA performance for this exact all-ones configuration has been observed.**
- New tests are provided in evidence_on_e/tests/test_full_unity.py, but have **not been executed in this session**; test success must be reported by the actual execution host.
- Earlier 19 CPU tests for the unchanged default E-based implementation were reported in the prior commit; changing the core and runner warrants rerunning both suites.

Before GPU training:
    python -m pytest -q evidence_on_e/tests/test_on_e.py evidence_on_e/tests/test_full_unity.py

On execution host, record the exact git commit, GPU1 UUID, full console output, test results, data and initial-state hashes; then execute experiments/full_unity/run_full_unity.sh.

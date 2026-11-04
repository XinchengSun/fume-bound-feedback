from types import SimpleNamespace
import numpy as np
import pytest
from fume_control import (BoundBracket, Evaluation, RiskGuard, decide,
                          ProtocolMismatch, bracket_from_bound_results, protocol_id)


def evaluation(lo, hi=None, risks=(0.1, 0.1), name='reference-v1', risk_id='ordered-risk-v1'):
    return Evaluation(BoundBracket(lo, lo if hi is None else hi, name, True), tuple(risks), risk_id)


def test_reported_epoch83_scalar_regression_not_real_checkpoint_replay():
    # Reported gain is used as a translated objective difference. Unknown
    # baseline objective and 200 individual risks are NOT fabricated as real data.
    old_risk = np.full(200, 0.001)
    delta = np.full(200, (-200*2.386e-7 - 8.88e-7) / 199)
    delta[0] = 8.88e-7
    old = evaluation(0.0, risks=old_risk)
    new = evaluation(0.0002361, risks=old_risk + delta)
    d = decide(old, new, RiskGuard(tuple([1e-5]*200)), min_gain=1e-8,
               old_dual_hint=-0.0029349)
    assert d.accepted
    assert d.guaranteed_gain == pytest.approx(0.0002361)
    assert d.mean_risk_change == pytest.approx(-2.386e-7)
    assert d.max_budget_fraction == pytest.approx(0.0888)


def test_positive_old_certificate_cannot_rescue_bad_fresh_bound():
    d = decide(evaluation(.4), evaluation(.39), RiskGuard((0, 0)),
               min_gain=1e-8, old_dual_hint=10.)
    assert not d.accepted and 'fresh_bound_gain_not_resolved' in d.reasons


def test_difference_of_lower_endpoints_is_not_used_as_gain_proof():
    d = decide(evaluation(.4, .45), evaluation(.42, .46), RiskGuard((0, 0)), min_gain=0)
    assert d.guaranteed_gain == pytest.approx(-.03)
    assert not d.accepted


def test_insufficient_numerical_gain_not_accepted():
    d = decide(evaluation(.4), evaluation(.4+1e-10), RiskGuard((0,0)), min_gain=1e-8)
    assert not d.accepted


def test_risk_tradeoffs_allowed_but_fixed_budget_checked():
    d = decide(evaluation(.4, risks=(.1,.1)), evaluation(.41, risks=(.11,.08)),
               RiskGuard((.015,.015)), min_gain=1e-8)
    assert d.accepted


def test_component_budget_violation_is_not_hidden_by_mean():
    d = decide(evaluation(.4), evaluation(.41, risks=(.12,.07)),
               RiskGuard((.01,.01)), min_gain=1e-8)
    assert not d.accepted and 'component_risk_budget_exceeded' in d.reasons


def test_mean_risk_budget_violation_rejects():
    d = decide(evaluation(.4), evaluation(.41, risks=(.11,.11)),
               RiskGuard((.02,.02)), min_gain=1e-8)
    assert not d.accepted and 'mean_risk_budget_exceeded' in d.reasons


def test_cumulative_anchor_is_not_reset_by_each_retry():
    guard = RiskGuard((.02,.02), anchor_risks=(.1,.1), anchor_component_budget=(.015,.015))
    d = decide(evaluation(.4, risks=(.11,.1)), evaluation(.41, risks=(.12,.08)), guard, min_gain=0)
    assert not d.accepted and 'cumulative_risk_budget_exceeded' in d.reasons


@pytest.mark.parametrize('field', ['objective','risk_id','dimension'])
def test_comparison_protocol_changes_are_fatal(field):
    old = evaluation(.4)
    if field == 'objective': new = evaluation(.41, name='other_events')
    elif field == 'risk_id': new = evaluation(.41, risk_id='other_risk_order')
    else: new = evaluation(.41, risks=(.1,))
    with pytest.raises(ProtocolMismatch):
        decide(old, new, RiskGuard((0,0)), min_gain=0)


@pytest.mark.parametrize('lo,hi,checked', [(float('nan'),.4,True),(.5,.4,True),(.3,.4,False)])
def test_bad_brackets_rejected(lo, hi, checked):
    with pytest.raises(ValueError): BoundBracket(lo,hi,'id',checked).validate()


def test_legacy_solver_adapter_has_two_sided_padding():
    results = [SimpleNamespace(primal=.500001,dual_lower=.5,max_violation=1e-12),
               SimpleNamespace(primal=.600002,dual_lower=.6,max_violation=1e-12)]
    r = bracket_from_bound_results(results,[1,1],objective_id='fixed')
    assert r.lower == pytest.approx(.55-1e-8)
    assert r.upper == pytest.approx(.5500015+1e-8)


@pytest.mark.parametrize('item', [SimpleNamespace(primal=.5,dual_lower=.4,max_violation=0),
                                  SimpleNamespace(primal=.5,dual_lower=.5,max_violation=.1)])
def test_solver_error_does_not_become_acceptance(item):
    with pytest.raises(ValueError):
        bracket_from_bound_results([item],[1],objective_id='fixed')


def test_protocol_digest_changes_with_event_and_kappa():
    x = {'events':[1,2], 'weights':[.5,.5], 'kappa':.01, 'data_sha':'abc'}
    assert protocol_id(x) == protocol_id(dict(reversed(list(x.items()))))
    assert protocol_id(x) != protocol_id({**x, 'kappa':.001})
    assert protocol_id(x) != protocol_id({**x, 'events':[2,1]})

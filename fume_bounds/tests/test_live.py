import numpy as np
import pytest
import torch
from live_reference import LiveReference, Event, RiskSystem, phase_one


def make_problem(n=8,kappa=.005,events=None,alpha=(.5,.5),task='multilabel',labels=None):
    if labels is None: labels=np.ones((n,1))
    return LiveReference(labels,anchor_alpha=alpha,
                         events=events or [Event('image',0,0)],kappa=kappa,
                         sample_ids=[str(i) for i in range(len(labels))],
                         heads_fingerprint='toy-heads-fixed',task=task,gap_tol=1e-6)


def bank(theta,n=8):
    v=torch.sigmoid(theta+torch.tensor([0.,.7],dtype=theta.dtype))
    i=v[None,:,None].expand(n,-1,1)
    t=torch.full_like(i,.6) + theta*0.0
    return i,t


def test_anchor_slack_is_kappa():
    p=make_problem()
    i,t=bank(torch.tensor(.7,dtype=torch.float64))
    s=p.solve(i.numpy(),t.numpy())
    np.testing.assert_allclose(s.witness_slack,p.kappa,rtol=0,atol=1e-14)
    assert s.objective_lower<=s.objective_upper


def test_old_empty_new_feasible():
    n=8;p=make_problem(n=n,kappa=.01)
    old_i=np.tile(np.array([.2,.4])[None,:,None],(n,1,1))
    new_i=np.tile(np.array([.8,.9])[None,:,None],(n,1,1))
    text=np.full_like(new_i,.6)
    current=RiskSystem.from_predictions(new_i,text,p.labels).risks(p.anchor_alpha)
    audit=phase_one(RiskSystem.from_predictions(old_i,text,p.labels),current+p.kappa)
    assert audit.status=='infeasible_certificate'
    assert audit.minimax_lower>.32
    s=p.solve(new_i,text)
    assert np.min(s.witness_slack)>.009999
    assert s.objective_lower<=1-np.sqrt(.0325)<=s.objective_upper+1e-9


def test_stale_start_not_empty():
    p=make_problem(kappa=.01)
    pi=np.tile(np.array([.2,.8])[None,:,None],(8,1,1));pt=np.full_like(pi,.6)
    risk=RiskSystem.from_predictions(pi,pt,p.labels)
    res=phase_one(risk,[.1,.2],start=[1.,0.])
    assert res.start_violation>.5
    assert res.status=='feasible'
    assert res.max_violation<=2e-8


def test_full_envelope_matches_finite_difference():
    p=make_problem()
    theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    i,t=bank(theta);s=p.solve(i.detach().numpy(),t.detach().numpy())
    loss=p.loss(i,t,s);loss.backward();actual=-theta.grad.item()
    eps=1e-4
    def value(x):
        ii,tt=bank(torch.tensor(x,dtype=torch.float64))
        ss=p.solve(ii.numpy(),tt.numpy())
        return ss.objective_upper
    fd=(value(.7+eps)-value(.7-eps))/(2*eps)
    assert actual==pytest.approx(fd,rel=1e-4,abs=2e-6)
    assert s.dual_weights.sum()>0


def test_zero_duals_still_have_event_gradient():
    p=make_problem(kappa=1.)
    theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    i,t=bank(theta);s=p.solve(i.detach().numpy(),t.detach().numpy())
    assert np.max(s.dual_weights)<1e-7
    loss=p.loss(i,t,s);loss.backward()
    assert -theta.grad.item()>0.1


def test_reject_old_snapshot_when_point_changed():
    p=make_problem();theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    i,t=bank(theta);s=p.solve(i.detach().numpy(),t.detach().numpy())
    ii,tt=bank(theta+.1)
    with pytest.raises(ValueError,match='Gradient point differs'):p.loss(ii,tt,s)
    assert torch.isfinite(p.loss(ii,tt,s,verify_snapshot_values=False))


def test_different_protocol_is_not_comparable():
    p=make_problem();p2=make_problem(kappa=.02)
    theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True);i,t=bank(theta)
    s=p.solve(i.detach().numpy(),t.detach().numpy())
    with pytest.raises(ValueError,match='different method'):p2.loss(i,t,s)


def test_reported_current_risks_must_match_anchor():
    p=make_problem();i,t=bank(torch.tensor(.7,dtype=torch.float64))
    with pytest.raises(ValueError,match='differs from anchor'):
        p.solve(i.numpy(),t.numpy(),reported_current_risks=np.zeros(2))


def test_no_detached_training_loss():
    p=make_problem();i,t=bank(torch.tensor(.7,dtype=torch.float64))
    s=p.solve(i.numpy(),t.numpy())
    with pytest.raises(ValueError,match='Detached'):p.loss(i,t,s)


def test_absent_event_orientation():
    p=make_problem(kappa=1.,events=[Event('image',0,0,target=0)])
    theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True);i,t=bank(theta)
    s=p.solve(i.detach().numpy(),t.detach().numpy())
    assert s.objective_upper==pytest.approx(1-i[0,1,0].item())
    p.loss(i,t,s).backward()
    assert theta.grad.item()>0


def test_multiclass_shape_sum_and_order():
    y=np.eye(3)[[0,1,2,0]]
    rng=np.random.default_rng(3)
    p1=rng.uniform(.1,1,(4,3,3));p1/=p1.sum(2,keepdims=True)
    p2=rng.uniform(.1,1,(4,3,3));p2/=p2.sum(2,keepdims=True)
    p=make_problem(labels=y,kappa=.01,alpha=(1/3,)*3,task='multiclass',
                   events=[Event('image',0,0),Event('text',1,1)])
    s=p.solve(p1,p2)
    assert len(s.anchor_risks)==6
    np.testing.assert_allclose(s.witness_slack,.01)
    with pytest.raises(ValueError,match='sum to one'):p.solve(p1*.9,p2)


@pytest.mark.parametrize('bad',[0.,-1.,float('nan')])
def test_bad_kappa_rejected(bad):
    with pytest.raises(ValueError):make_problem(kappa=bad)


def test_pytorch_head_parameters_frozen_but_encoder_gradient_flows():
    torch.manual_seed(2)
    enc=torch.nn.Linear(3,2,dtype=torch.float64)
    heads=torch.nn.ModuleList([torch.nn.Linear(2,1,dtype=torch.float64) for _ in range(2)])
    for h in heads:
        h.requires_grad_(False);h.eval()
    x=torch.randn(8,3,dtype=torch.float64)
    i=torch.stack([torch.sigmoid(h(enc(x))) for h in heads],dim=1)
    t=i*.7+.1
    p=make_problem(kappa=.02);s=p.solve(i.detach().numpy(),t.detach().numpy())
    p.loss(i,t,s).backward()
    assert enc.weight.grad.norm().item()>1e-8
    assert all(v.grad is None for h in heads for v in h.parameters())


def test_probability_space_replay_gradient_matches_full_graph():
    p=make_problem();theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    i,t=bank(theta);s=p.solve(i.detach().numpy(),t.detach().numpy())
    gi,gt=p.coefficients(i,t,s)
    g1=torch.autograd.grad(p.loss(i,t,s),theta,retain_graph=True)[0]
    g2=torch.autograd.grad((i*gi).sum()+(t*gt).sum(),theta)[0]
    assert torch.allclose(g1,g2,atol=1e-12,rtol=1e-10)


def test_v3_reference_adapter():
    p=make_problem();i,t=bank(torch.tensor(.7,dtype=torch.float64))
    s=p.solve(i.numpy(),t.numpy());r=s.reference_solve();ev=r.evaluation()
    assert ev.bound.checked
    assert ev.bound.objective_id==p.objective_id
    assert len(ev.risks)==2


def test_anchor_can_be_on_simplex_boundary():
    p=make_problem(alpha=(1.,0.));i,t=bank(torch.tensor(.7,dtype=torch.float64))
    s=p.solve(i.numpy(),t.numpy())
    assert np.min(s.witness_slack)>0


def test_both_encoder_branches_receive_gradient():
    p=make_problem(kappa=.04,events=[Event('image',0,0),Event('text',1,0)])
    theta=torch.tensor([.3,.5],dtype=torch.float64,requires_grad=True)
    i,_=bank(theta[0]);t,_=bank(theta[1]);s=p.solve(i.detach().numpy(),t.detach().numpy())
    p.loss(i,t,s).backward()
    assert torch.all(theta.grad.abs()>1e-6)


def test_probability_one_half_not_declared_certain():
    p=make_problem(kappa=.01)
    i=np.full((8,2,1),.5);s=p.solve(i,i)
    assert s.objective_upper==pytest.approx(.5)
    assert s.objective_lower<=.5
    # This tests output semantics, NOT preservation of population aleatoric uncertainty.

from types import SimpleNamespace
import pytest
import torch
from fume_control import *


def test_existing_boundresult_fields_adapter_trains_live_encoder():
    m=torch.nn.Linear(1,1,dtype=torch.float64)
    with torch.no_grad():m.weight.zero_(); m.bias.zero_()
    opt=torch.optim.Adam(m.parameters(),lr=.02)
    c=TransactionalController(m,opt)
    x=torch.ones(2,1,dtype=torch.float64); y=torch.full_like(x,.8)
    def loss(batch): return ((m(batch[0])-batch[1])**2).mean()
    calls=[]
    def solve():
        r=float(loss((x,y)))
        calls.append(float(m.bias.detach()))
        # Analytic enclosure of this small toy's objective; not INRIA.
        br=SimpleNamespace(primal=1-r,dual_lower=1-r,max_violation=0.)
        return ReferenceSolve([br],[1.],[r],'fixed-protocol','risk-order',numerical_padding=1e-12)
    r=run_encoder_segment(c,replay_batches=[(x,y),(x,y)],current_loss=loss,
                          solve_current_reference=solve,risk_guard=RiskGuard((0.,)),round_id='bridge')
    assert r.accepted and len(calls)==2 and calls[0]!=calls[1]


def test_replay_batch_mutation_is_not_silently_accepted():
    m=torch.nn.Linear(1,1,dtype=torch.float64); opt=torch.optim.Adam(m.parameters())
    x=torch.ones(1,1,dtype=torch.float64)
    p=TorchSegment(m,opt,[x],lambda b:m(b).square().mean())
    x.add_(1)
    with pytest.raises(ValueError,match='modified'):
        p(AttemptSpec(0,1.,(.001,),1,'test'))


def test_explicit_frozen_submodule_training_mode_is_preserved():
    m=torch.nn.Sequential(torch.nn.Linear(1,1,dtype=torch.float64),torch.nn.Dropout(.5))
    opt=torch.optim.Adam(m.parameters(),lr=.1)
    def mode(): m.train(); m[1].eval()
    p=TorchSegment(m,opt,[torch.ones(1,1,dtype=torch.float64)],
                   lambda b:m(b).square().mean(),set_training_mode=mode)
    p(AttemptSpec(0,1.,(.1,),1,'test'))
    assert m.training and not m[1].training

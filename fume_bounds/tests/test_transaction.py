import copy
import random
import numpy as np
import pytest
import torch
from fume_control import *
from fume_control.state import Snapshot, digest, RNGState


class Scalar(torch.nn.Module):
    def __init__(self, value=0., dtype=torch.float64):
        super().__init__()
        self.q = torch.nn.Parameter(torch.tensor(value, dtype=dtype))
        self.register_buffer('running', torch.tensor(2., dtype=dtype))
        self.W = torch.tensor([3.,4.], dtype=dtype)  # original FUME-like ordinary tensor


def state(model, opt):
    return digest({'m':model.state_dict(), 'W':model.W, 'o':opt.state_dict()})


def oracle(model, target=.6, name='synthetic'):
    r = float((model.q.detach()-target)**2)
    return Evaluation(BoundBracket(1-r,1-r,name,True), (r,), 'one-risk')


def proposal(model, opt, spec, target=.6):
    for _ in range(spec.optimizer_steps):
        opt.zero_grad(set_to_none=True)
        loss = (model.q-target).square()
        loss.backward(); opt.step()
    return ProposalInfo(old_dual_hint=-.0029349)


def test_reject_then_smaller_actual_adam_trajectory_commits_consistent_state(tmp_path):
    m = Scalar(); opt = torch.optim.Adam(m.parameters(), lr=2.)
    control = TransactionalController(m,opt,ControlConfig(max_attempts=4),log_path=tmp_path/'trace.jsonl')
    report = control.run_segment(lambda s: proposal(m,opt,s),lambda:oracle(m),RiskGuard((0,)),
                                 optimizer_steps=1,round_id='replay')
    assert report.accepted and len(report.attempts)==2
    assert report.attempted_optimizer_steps==2 and report.committed_optimizer_steps==1
    assert report.attempts[0]['learning_rates']==[2.]
    assert report.attempts[1]['learning_rates']==[1.]
    direct = Scalar(); direct_opt = torch.optim.Adam(direct.parameters(),lr=1.)
    proposal(direct,direct_opt,AttemptSpec(0,1.,(1.,),1,'manual'))
    assert state(m,opt) == state(direct,direct_opt)
    assert int(opt.state[m.q]['step'])==1
    assert report.decision.old_hint < 0


def test_all_rejections_restore_parameters_moments_rng_and_W():
    random.seed(8); np.random.seed(8); torch.manual_seed(8)
    m = Scalar(.2); opt = torch.optim.Adam(m.parameters(),lr=.1)
    proposal(m,opt,AttemptSpec(0,1.,(.1,),1,'warmup'))
    before = state(m,opt); rng = RNGState.capture()
    controller = TransactionalController(m,opt,ControlConfig(max_attempts=3))
    draws=[]
    def bad(s):
        draws.append((random.random(),float(np.random.rand()),float(torch.rand(()))))
        m.W.add_(1); m.running.add_(1)
        for _ in range(s.optimizer_steps):
            opt.zero_grad(set_to_none=True); (-m.q).backward(); opt.step()
        return ProposalInfo()
    # Fixed zero objective: nothing can show a positive bound improvement.
    def evaluate(): return Evaluation(BoundBracket(0,0,'fixed',True),(0.,),'risk')
    report = controller.run_segment(bad,evaluate,RiskGuard((0,)),optimizer_steps=1)
    assert not report.accepted and len(report.attempts)==3
    assert state(m,opt)==before
    assert draws[0]==draws[1]==draws[2]
    after=RNGState.capture()
    assert digest(after.__dict__)==digest(rng.__dict__)
    assert report.committed_optimizer_steps==0


def test_failed_controller_will_not_replay_the_same_failed_epoch_forever():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1)
    c=TransactionalController(m,opt,ControlConfig(max_attempts=2))
    ev=lambda:Evaluation(BoundBracket(0,0,'same',True),(0.,),'same')
    r=c.run_segment(lambda s:proposal(m,opt,s),ev,RiskGuard((0,)),optimizer_steps=1,round_id='83')
    assert not r.accepted
    with pytest.raises(ControllerStopped):
        c.run_segment(lambda s:proposal(m,opt,s),ev,RiskGuard((0,)),optimizer_steps=1,round_id='84')
    saved=c.state_dict()
    other=TransactionalController(m,opt,ControlConfig(max_attempts=2)); other.load_state_dict(saved)
    with pytest.raises(ControllerStopped): other.run_segment(None,None,None)


def test_fp32_zero_parameter_motion_does_not_commit_30_adam_steps():
    m=Scalar(1.,torch.float32); opt=torch.optim.Adam(m.parameters(),lr=1e-20)
    before=state(m,opt)
    c=TransactionalController(m,opt,ControlConfig(min_lr=1e-30))
    r=c.run_segment(lambda s:proposal(m,opt,s,target=0),lambda:oracle(m,target=0),
                    RiskGuard((0,)),optimizer_steps=30)
    assert not r.accepted and r.stop_reason=='no_effective_parameter_update'
    assert r.attempted_optimizer_steps==30 and r.committed_optimizer_steps==0
    assert state(m,opt)==before and len(opt.state)==0


def test_effective_parameter_floor_checked_after_actual_write():
    m=Scalar(0.); opt=torch.optim.SGD(m.parameters(),lr=1e-11)
    c=TransactionalController(m,opt,ControlConfig(parameter_atol=1e-8,min_lr=1e-20))
    r=c.run_segment(lambda s:proposal(m,opt,s),lambda:oracle(m),RiskGuard((0,)),optimizer_steps=1)
    assert not r.accepted and r.stop_reason=='no_effective_parameter_update'
    assert float(m.q.detach())==0.


def test_learning_rate_overwrite_is_detected_and_rolled_back():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1)
    initial=state(m,opt)
    c=TransactionalController(m,opt)
    def broken(s):
        opt.param_groups[0]['lr']=.77
        return proposal(m,opt,s)
    with pytest.raises(ContractError, match='overwritten'):
        c.run_segment(broken,lambda:oracle(m),RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==initial


def test_wrong_actual_step_count_fails_instead_of_faking_30_steps():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1); initial=state(m,opt)
    c=TransactionalController(m,opt)
    with pytest.raises(ContractError,match='Expected 30'):
        c.run_segment(lambda s:proposal(m,opt,AttemptSpec(0,1.,(.1,),1,'wrong')),
                      lambda:oracle(m),RiskGuard((0,)),optimizer_steps=30)
    assert state(m,opt)==initial


def test_evaluator_cannot_silently_mutate_parameters():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1); initial=state(m,opt)
    c=TransactionalController(m,opt)
    def evil():
        m.q.add_(.1)
        return oracle(m)
    with pytest.raises(ContractError,match='evaluate'):
        c.run_segment(lambda s:proposal(m,opt,s),evil,RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==initial


def test_runtime_exception_restores_full_state():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1); initial=state(m,opt)
    c=TransactionalController(m,opt)
    def crash(s):
        proposal(m,opt,s); m.W.add_(2)
        raise RuntimeError('deliberate failure')
    with pytest.raises(RuntimeError,match='deliberate'):
        c.run_segment(crash,lambda:oracle(m),RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==initial


def test_reference_protocol_cannot_change_halfway():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1); initial=state(m,opt)
    c=TransactionalController(m,opt)
    def eval_new():
        return oracle(m,name='old' if float(m.q.detach())==0 else 'new-kappa')
    with pytest.raises(ProtocolMismatch):
        c.run_segment(lambda s:proposal(m,opt,s),eval_new,RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==initial


def test_batchnorm_buffers_nonpersistent_buffers_modes_and_extras_restore():
    m=Scalar(); m.register_buffer('temp',torch.tensor(5.),persistent=False)
    m.extra=torch.nn.Dropout(.5); m.extra.eval()
    opt=torch.optim.Adam(m.parameters(),lr=.1)
    generator=torch.Generator().manual_seed(77)
    extras={'generator':GeneratorState(generator)}
    gen_before=generator.get_state().clone()
    c=TransactionalController(m,opt,ControlConfig(max_attempts=1),stateful=extras)
    def go(s):
        m.train(); m.temp.add_(1); torch.rand(4,generator=generator)
        return proposal(m,opt,s)
    constant=lambda:Evaluation(BoundBracket(0,0,'fixed',True),(0.,),'risk')
    r=c.run_segment(go,constant,RiskGuard((0,)),optimizer_steps=1)
    assert not r.accepted and float(m.temp)==5
    assert not m.extra.training and torch.equal(generator.get_state(),gen_before)


def test_posthoc_parameter_interpolation_is_detected():
    m=Scalar(); opt=torch.optim.SGD(m.parameters(),lr=.1); before=state(m,opt)
    c=TransactionalController(m,opt,ControlConfig(max_attempts=8))
    def interpolate(s):
        proposal(m,opt,s)
        with torch.no_grad(): m.q.mul_(5.96e-8)  # former failure pattern
    with pytest.raises(ContractError,match='after the last'):
        c.run_segment(interpolate,lambda:oracle(m),RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==before


def test_posthoc_optimizer_replacement_is_detected():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1); before=state(m,opt)
    c=TransactionalController(m,opt)
    def overwrite(s):
        proposal(m,opt,s)
        opt.state[m.q]['step'].add_(30)
    with pytest.raises(ContractError,match='after the last'):
        c.run_segment(overwrite,lambda:oracle(m),RiskGuard((0,)),optimizer_steps=1)
    assert state(m,opt)==before


def test_torch_segment_runs_real_losses_and_two_parameters():
    m=torch.nn.Linear(2,1,dtype=torch.float64)
    torch.nn.init.zeros_(m.weight); torch.nn.init.zeros_(m.bias)
    opt=torch.optim.Adam(m.parameters(),lr=.01)
    batches=[(torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64),
              torch.tensor([[.8],[.6]],dtype=torch.float64))]*2
    def loss(batch):
        x,y=batch; return ((m(x)-y)**2).mean()
    def evaluate():
        r=float(loss(batches[0])); return Evaluation(BoundBracket(1-r,1-r,'toy',True),(r,),'r')
    ctl=TransactionalController(m,opt)
    r=ctl.run_segment(TorchSegment(m,opt,batches,loss),evaluate,RiskGuard((0,)),optimizer_steps=2)
    assert r.accepted and r.committed_optimizer_steps==2
    assert r.attempts[-1]['changed_coordinates']==3


def test_scheduler_state_and_optimizer_lr_restore_on_rejection():
    m=Scalar(); opt=torch.optim.Adam(m.parameters(),lr=.1)
    scheduler=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9)
    snapshot=copy.deepcopy(scheduler.state_dict())
    c=TransactionalController(m,opt,ControlConfig(max_attempts=1),stateful={'scheduler':scheduler})
    def p(s):
        out=proposal(m,opt,s); scheduler.step(); return out
    ev=lambda:Evaluation(BoundBracket(0,0,'same',True),(0.,),'r')
    with pytest.raises(ContractError,match='after the last'):
        c.run_segment(p,ev,RiskGuard((0,)),optimizer_steps=1)
    assert scheduler.state_dict()==snapshot and opt.param_groups[0]['lr']==.1

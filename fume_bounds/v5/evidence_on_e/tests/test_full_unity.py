"""Regression tests for user-specified five-term L_total with ALL ACTIVE WEIGHTS = 1."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from core import EvidenceOnE, AuxConfig, objective, base
from run import parser


def make_model(task='matching'):
    torch.manual_seed(2)
    return EvidenceOnE(5,6,4,layers=1,task=task)


def make_batch():
    g=torch.Generator().manual_seed(14)
    return (torch.randn(7,5,generator=g),torch.randn(7,6,generator=g),
            F.one_hot(torch.arange(7)%4,4).float())


def test_historical_default_CL005_exact_loss_and_gradients():
    old=make_model('none')
    new=copy.deepcopy(old)
    batch=make_batch()
    total,sup,cl=base.minimal_objective(old,*batch)
    total.backward()
    aux=AuxConfig(vicinal=0,noise=0,error_kl=0)
    actual,logs,_=objective(new,batch,aux,backward=True)
    assert torch.equal(total.detach(),actual)
    assert logs['contrastive_weight']==pytest.approx(.05)
    for p,q in zip(old.parameters(),new.parameters()):
        assert torch.equal(p.grad,q.grad)


def test_CL1_one_time_only_and_gradient_difference():
    old=make_model('none')
    full=copy.deepcopy(old)
    batch=make_batch()
    e,sup,cl=base.minimal_objective(old,*batch)
    aux=AuxConfig(vicinal=0,noise=0,error_kl=0)
    actual,logs,probe=objective(full,batch,aux,cl_weight=1,backward=True,probe=True)
    assert actual.item()==pytest.approx((sup+cl).item(),abs=1e-6)
    assert actual.item()-e.item()==pytest.approx(.95*cl.item(),abs=1e-6)
    assert logs['weighted_contrastive']==pytest.approx(logs['contrastive'],abs=1e-6)
    assert logs['supervision']==pytest.approx(sup.item(),abs=1e-6)
    assert logs['total']==pytest.approx(logs['supervision']+logs['contrastive'],abs=1e-6)
    assert probe['image']['contrastive']['weighted_grad_norm']>0
    torch.autograd.backward([sup,cl])
    for p,q in zip(old.parameters(),full.parameters()):
        assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_all_five_weights_one_no_kl_and_nonzero_pair_gradients():
    m=make_model()
    c=AuxConfig(vicinal=1,noise=1,error_kl=0,pair_clean_weight=1,pair_chunk=3)
    torch.manual_seed(8)
    actual,logs,probe=objective(m,make_batch(),c,(.5,.7),epoch=10,
                                cl_weight=1,backward=True,probe=True)
    assert logs['contrastive_weight']==1
    assert logs['kl_coefficient']==0
    assert 'error_kl' not in logs
    for k in ('supervision','contrastive','pair_clean','vicinal','noise'):
        assert k in logs and np.isfinite(logs[k])
    assert logs['total']==pytest.approx(sum(logs[k] for k in
        ('supervision','contrastive','pair_clean','vicinal','noise')),abs=2e-5)
    assert logs['weighted_pair_clean']==pytest.approx(logs['pair_clean'],abs=1e-6)
    assert logs['weighted_vicinal']==pytest.approx(logs['vicinal'],abs=1e-6)
    assert logs['weighted_noise']==pytest.approx(logs['noise'],abs=1e-6)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    assert any(p.grad is not None and p.grad.norm().item()>0
               for n,p in m.named_parameters() if 'pair_head' in n)
    assert all(probe[side][term]['weighted_grad_norm']>=0 for side in ('image','text')
               for term in ('supervision','contrastive','pair_clean','vicinal','noise'))
    assert 'W' not in dict(m.named_parameters())


def test_cl_weight_rejects_nonfinite_and_negative():
    m=make_model('none')
    batch=make_batch()
    for c in (-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            objective(m,batch,AuxConfig(vicinal=0,noise=0,error_kl=0),cl_weight=c)


def test_cli_parses_explicit_CL1_unchanged_default():
    empty=parser()
    assert empty.get_default('cl_weight') is None  # DEFAULTS resolved inside main
    args=empty.parse_args(['--data','example.npz','--output','new','--cl-weight','1',
                           '--pair-clean-weight','1','--vicinal','1',
                           '--noise','1','--error-kl','0'])
    assert args.cl_weight==1 and args.pair_clean_weight==1 and args.error_kl==0


def test_one_epoch_cpu_cli_has_five_terms(tmp_path):
    rng=np.random.default_rng(4)
    p=tmp_path/'small.npz'
    arrays={}
    for split,n in [('train',7),('val',5),('test',6)]:
        arrays[split+'_image']=rng.normal(size=(n,5)).astype('float32')
        arrays[split+'_text']=rng.normal(size=(n,6)).astype('float32')
        arrays[split+'_labels']=np.eye(4,dtype='float32')[np.arange(n)%4]
    np.savez(p,**arrays)
    out=tmp_path/'trial'
    cmd=[sys.executable,str(ROOT/'run.py'),'--data',str(p),'--output',str(out),
         '--classes','4','--layers','1','--batch-size','3','--epochs','1',
         '--task','matching','--device','cpu','--threads','1',
         '--cl-weight','1','--pair-clean-weight','1','--vicinal','1',
         '--noise','1','--error-kl','0','--uq-pairs','10','--final-test']
    run=subprocess.run(cmd,text=True,capture_output=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    cfg=json.loads((out/'provenance.json').read_text())
    assert cfg['contrastive_weight']==1
    assert cfg['fixed_E_membership_supervision'] is True
    assert cfg['fixed_E_clean_objective'] is False
    history=[json.loads(x) for x in (out/'history.jsonl').read_text().splitlines()]
    assert len(history)==1
    h=history[0]
    assert h['kl_coefficient']==0
    assert h['weighted_contrastive']==pytest.approx(h['contrastive'],abs=1e-5)
    assert h['total']==pytest.approx(h['supervision']+h['contrastive']+
                  h['pair_clean']+h['vicinal']+h['noise'],abs=2e-5)
    assert json.loads((out/'test_metrics.json').read_text())['queries']==6

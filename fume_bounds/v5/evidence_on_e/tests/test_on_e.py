"""CPU regression suite. Synthetic checks are not INRIA/GPU performance claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import EvidenceOnE,AuxConfig,objective,base,LEGACY_PATH,sample_pairs,mixup,augment,expected_ce,wrong_kl
from metrics import statistics,error_auroc,evidence_report

torch.set_num_threads(1)


def model(task='matching'):
    torch.manual_seed(2)
    return EvidenceOnE(5,6,4,layers=1,task=task)


def batch():
    gen=torch.Generator().manual_seed(7)
    return (torch.randn(7,5,generator=gen),torch.randn(7,6,generator=gen),
            F.one_hot(torch.arange(7)%4,4).float())


@pytest.mark.parametrize('task',['none','category','matching'])
def test_backbone_and_rng_identical_to_original(task):
    torch.manual_seed(2)
    legacy=base.load_module(LEGACY_PATH,'_test_original')
    original=legacy.FUME('cpu',5,6,4,4,1)
    w=original.W
    del original.W
    original.register_buffer('W',w)
    state=torch.get_rng_state().clone()
    m=model(task)
    assert torch.equal(torch.get_rng_state(),state)
    assert base.backbone_hash(m)==base.backbone_hash(original)
    x,t,_=batch()
    assert torch.equal(m.img_net(x),original.img_net(x))
    assert torch.equal(m.text_net(t),original.text_net(t))
    assert ('pair_head' in dict(m.named_children()))==(task=='matching')
    assert ('img_evidence' in dict(m.named_children()))==(task=='category')


@pytest.mark.parametrize('task',['none','category','matching'])
def test_disabled_aux_is_exact_E_loss_gradient_update(task):
    m=model(task)
    other=copy.deepcopy(m)
    b=batch()
    total,sup,cl=base.minimal_objective(m,*b)
    total.backward()
    result,_,_=objective(other,b,AuxConfig(vicinal=0,noise=0,error_kl=0),
                         backward=True,probe=True)
    assert torch.equal(total.detach(),result)
    for (n,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            assert torch.equal(p.grad,q.grad),n
    for obj in (m,other):
        torch.optim.Adam(obj.parameters(),lr=5e-5,betas=(.5,.999)).step()
    for k,v in m.state_dict().items():
        assert torch.equal(v,other.state_dict()[k]),k


@pytest.mark.parametrize('task',['category','matching'])
def test_joint_and_staged_gradients_and_no_BCE(task,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('BCE must not be called')
    monkeypatch.setattr(F,'binary_cross_entropy_with_logits',forbidden)
    m=model(task)
    other=copy.deepcopy(m)
    cfg=AuxConfig(pair_chunk=3)
    torch.manual_seed(34)
    a,_,_=objective(m,batch(),cfg,(.7,.8),epoch=10)
    a.backward()
    torch.manual_seed(34)
    b,logs,probe=objective(other,batch(),cfg,(.7,.8),epoch=10,
                           backward=True,probe=True)
    assert a.item()==pytest.approx(b.item(),rel=1e-7)
    assert all(k in logs for k in ('supervision','contrastive','vicinal','noise','error_kl'))
    for (n,p),(_,q) in zip(m.named_parameters(),other.named_parameters()):
        assert p.grad is not None and q.grad is not None,n
        assert torch.isfinite(q.grad).all()
        assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=2e-5),n
    assert all(probe[s]['supervision']['weighted_grad_norm']>0 for s in ('image','text'))
    assert 'W' not in dict(m.named_parameters())


def test_pair_chunk_does_not_reweight_small_tail():
    m=model()
    other=copy.deepcopy(m)
    torch.manual_seed(33)
    a,_,_=objective(m,batch(),AuxConfig(pair_chunk=3),epoch=10)
    a.backward()
    torch.manual_seed(33)
    b,_,_=objective(other,batch(),AuxConfig(pair_chunk=100),epoch=10)
    b.backward()
    assert a.item()==pytest.approx(b.item(),rel=1e-6)
    for p,q in zip(m.parameters(),other.parameters()):
        assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=3e-5)


def test_overlap_labels_and_no_negative_edge_case():
    y=batch()[2]
    torch.manual_seed(3)
    i,j,t,count=sample_pairs(y,AuxConfig())
    assert torch.equal((y[i]*y[j]).sum(1)>0,t[:,0].bool())
    assert count['pair_count']==14 and count['pair_positives']==7
    allsame=torch.ones(3,1)
    i,j,t,count=sample_pairs(allsame,AuxConfig())
    assert len(i)==3 and t.sum()==3 and count['anchors_without_negative']==3


def test_mix_and_noise_preserve_targets():
    y=torch.tensor([[1.,1.,0.,0.],[0.,1.,0.,1.],[1.,0.,0.,0.]])
    x,t,target=mixup(y,y,y,AuxConfig())
    assert torch.equal(x,t) and torch.equal(x,target)
    assert (target.sum(1)>1).any()
    x,t,target=augment(y,y,y,AuxConfig(),(0.,0.),'noise')
    assert torch.equal(x,t)
    for row in range(3):
        lam=x[row,y[row].bool()][0]
        assert torch.allclose(target[row],lam*y[row]+(1-lam)*.5)


def test_beta_formula_kl_and_prior():
    a=torch.tensor([[2.,3.,5.]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[4.,2.,6.]],dtype=torch.double,requires_grad=True)
    t=torch.tensor([[.6,.2,.8]])
    actual=expected_ce(dict(alpha=a,beta=b),t)
    direct=(torch.digamma(a+b)-t.double()*torch.digamma(a)-
            (1-t.double())*torch.digamma(b)).mean()
    assert torch.allclose(actual,direct,atol=1e-12)
    y=torch.tensor([[1.,0.,1.]])
    masked_a=torch.where(y.bool(),torch.ones_like(a),a)
    masked_b=torch.where(y.bool(),b,torch.ones_like(b))
    manual=torch.distributions.kl_divergence(
        torch.distributions.Beta(masked_a,masked_b),
        torch.distributions.Beta(torch.ones_like(a),torch.ones_like(b))).mean()
    assert torch.allclose(wrong_kl(dict(alpha=a,beta=b),y),manual,atol=1e-12)
    with pytest.raises(ValueError):
        wrong_kl(dict(alpha=a,beta=b),t)
    s=statistics(torch.ones(1,1),torch.ones(1,1))
    assert s['probability'].item()==.5 and s['vacuity'].item()==1
    assert s['aleatoric_entropy'].item()==pytest.approx(.5)


@pytest.mark.parametrize('task',['category','matching'])
def test_uq_does_not_change_rng_mode_or_parameters(task):
    m=model(task)
    state=torch.get_rng_state().clone()
    before=copy.deepcopy(m.state_dict())
    report=evidence_report(m,torch.utils.data.TensorDataset(*batch()),pairs=20,chunk=3)
    assert report['task']==task and m.training and torch.equal(torch.get_rng_state(),state)
    for k,v in before.items():
        assert torch.equal(v,m.state_dict()[k])
    for v in report['views'].values():
        assert 0<=v['brier']<=1 and np.isfinite(v['nll'])


def test_non_single_positive_rejected_by_E_and_tied_auroc():
    x,t,y=batch()
    y[0,1]=1
    with pytest.raises(ValueError):
        objective(model(),(x,t,y))
    assert error_auroc([0,1,0,1],[.5]*4)==.5
    assert error_auroc([0,1],[0,1])==1
    assert error_auroc([1,1],[0,1]) is None


def write_data(path):
    arrays={}
    gen=np.random.default_rng(3)
    for split,n in [('train',7),('val',5),('test',6)]:
        arrays[split+'_image']=gen.normal(size=(n,5)).astype('float32')
        arrays[split+'_text']=gen.normal(size=(n,6)).astype('float32')
        arrays[split+'_labels']=np.eye(4,dtype='float32')[np.arange(n)%4]
    np.savez(path,**arrays)


def run(*args):
    p=subprocess.run([sys.executable,str(ROOT/'run.py'),*map(str,args)],
                     capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout+p.stderr
    return p


@pytest.mark.parametrize('task',['none','category','matching'])
def test_cli_train_resume_same_as_continuous(tmp_path,task):
    data=tmp_path/'data.npz'
    write_data(data)
    common=['--data',data,'--classes',4,'--layers',1,'--batch-size',3,
            '--task',task,'--device','cpu','--threads',1,'--pair-chunk',2,'--uq-pairs',10]
    run(*common,'--output',tmp_path/'first','--epochs',1)
    run('--data',data,'--output',tmp_path/'resume','--resume',tmp_path/'first/last.pt',
        '--epochs',2,'--device','cpu','--threads',1,'--final-test')
    run(*common,'--output',tmp_path/'whole','--epochs',2,'--final-test')
    a=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    b=torch.load(tmp_path/'whole/last.pt',weights_only=True)
    for k,v in a['model'].items():
        assert torch.equal(v,b['model'][k]),k
    for k,v in a['optimizer']['state'].items():
        for key,val in v.items():
            assert torch.equal(val,b['optimizer']['state'][k][key])
    assert json.loads((tmp_path/'resume/test_metrics.json').read_text())['queries']==6
    assert json.loads((tmp_path/'whole/completion.json').read_text())['total_updates']==6
    rows=[json.loads(x) for x in (tmp_path/'whole/history.jsonl').read_text().splitlines()]
    assert len(rows)==2 and all(x['train_rows']==7 for x in rows)
    assert (tmp_path/'first/test_metrics.json').exists() is False


def test_no_overwrite_and_no_legacy_resume(tmp_path):
    data=tmp_path/'data.npz'
    write_data(data)
    out=tmp_path/'out'
    out.mkdir()
    (out/'keep').write_text('keep')
    p=subprocess.run([sys.executable,str(ROOT/'run.py'),
                      '--data',str(data),'--output',str(out)],
                     capture_output=True,text=True)
    assert p.returncode!=0 and 'new/empty' in p.stderr and (out/'keep').read_text()=='keep'
    torch.save({'format':'fume_minimal_no_necessity_v1'},tmp_path/'old.pt')
    p=subprocess.run([sys.executable,str(ROOT/'run.py'),
                      '--data',str(data),'--output',str(tmp_path/'new'),
                      '--resume',str(tmp_path/'old.pt')],
                     capture_output=True,text=True)
    assert p.returncode!=0 and 'Only evidence_on_e' in p.stderr

"""Synthetic/unit checks; no INRIA performance is asserted."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import minimal_control as m

torch.set_num_threads(1)


class TinyTower(nn.Module):
    def __init__(self, din, k):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(din,12),nn.Linear(12,k,bias=False)])
    def forward(self,x):
        h=self.layers[-1](F.relu(self.layers[0](x)))
        return h/h.norm(dim=1,keepdim=True)


class TinyFume(nn.Module):
    def __init__(self,device='cpu',image_dim=5,text_dim=6,k=4,classes=4,layers=1):
        super().__init__()
        self.img_net=TinyTower(image_dim,k);self.text_net=TinyTower(text_dim,k)
        self.W=nn.init.orthogonal_(torch.empty(k,k))
        self.W=self.W/self.W.norm(dim=1,keepdim=True)
    def forward(self,*args):
        raise AssertionError('Original FUME.forward/c MUST NOT be called')


def model():
    torch.manual_seed(5)
    net=TinyFume();w=net.W;del net.W;net.register_buffer('W',w)
    return net


def batch():
    torch.manual_seed(7)
    return torch.randn(7,5),torch.randn(7,6),F.one_hot(torch.arange(7)%4,4).float()


def test_direct_membership_objective_and_frozen_W():
    net=model();x,t,y=batch()
    total,sup,cl=m.minimal_objective(net,x,t,y)
    zi,zt=net.img_net(x),net.text_net(t)
    w=net.W/net.W.norm(dim=1,keepdim=True)
    expected=((zi@w).relu()-y).square().sum(1).sqrt().mean()+((zt@w).relu()-y).square().sum(1).sqrt().mean()
    assert torch.allclose(sup,expected)
    assert torch.allclose(total,sup+.05*cl)
    assert 'W' in net.state_dict() and 'W' not in dict(net.named_parameters())


@pytest.mark.parametrize('n',[1,2,7,12])
def test_original_cl_value_and_gradient_against_stable_form(n):
    torch.manual_seed(n)
    x=torch.randn(n,4,dtype=torch.float64,requires_grad=True)
    t=torch.randn(n,4,dtype=torch.float64,requires_grad=True)
    zi,zt=F.normalize(x,dim=1),F.normalize(t,dim=1)
    actual=m.original_contrastive(zi,zt)
    z=torch.cat([zi,zt]);scores=z@z.T
    scores=scores.masked_fill(torch.eye(2*n,dtype=torch.bool),-torch.inf)
    expected=2*F.cross_entropy(scores,(torch.arange(2*n)+n)%(2*n))
    assert torch.allclose(actual,expected,atol=1e-12)
    a=torch.autograd.grad(actual,(x,t),retain_graph=True)
    b=torch.autograd.grad(expected,(x,t))
    assert all(torch.allclose(u,v,atol=1e-12) for u,v in zip(a,b))


@pytest.mark.parametrize('kind',['multihot','soft','empty','nan'])
def test_reject_non_onehot_instead_of_changing_labels(kind):
    x,t,y=batch()
    if kind=='multihot':y[0,1]=1
    elif kind=='soft':y[0]=.25
    elif kind=='empty':y[0]=0
    else:y[0,0]=torch.nan
    with pytest.raises(ValueError):m.minimal_objective(model(),x,t,y)


def test_probe_does_not_write_parameter_gradients():
    net=model();loss,sup,cl=m.minimal_objective(net,*batch())
    w=net.W.clone();before={k:v.clone() for k,v in net.state_dict().items()}
    report=m.output_layer_probe(net,sup,cl)
    assert all(p.grad is None for p in net.parameters())
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
    assert report['image']['supervision_grad_norm']>0
    loss.backward()
    opt=torch.optim.Adam(net.parameters(),lr=5e-5,betas=(.5,.999));opt.step()
    for prefix in ['img_net.','text_net.']:
        assert any(not torch.equal(v,before[k]) for k,v in net.state_dict().items() if k.startswith(prefix))
    assert torch.equal(w,net.W)


def test_zero_residual_norm_has_finite_zero_gradient():
    delta=torch.zeros(3,4,requires_grad=True)
    torch.linalg.vector_norm(delta,dim=1).mean().backward()
    assert torch.equal(delta.grad,torch.zeros_like(delta))


def test_evaluation_all_queries_mode_rng_and_independent_AP():
    net=model();x,t,y=batch();state=torch.get_rng_state().clone()
    report=m.evaluate(net,TensorDataset(x,t,y),batch_size=3,chunk=2)
    assert net.training and torch.equal(state,torch.get_rng_state())
    with torch.no_grad():
        zi,zt=net.img_net(x),net.text_net(t)
        sim=(F.normalize(zi,dim=1)@F.normalize(zt,dim=1).T).numpy()
    rel=(y@y.T).numpy()>0
    def direct(s,r):
        vals=[]
        for i in range(len(s)):
            order=sorted(range(len(s)),key=lambda j:(-float(s[i,j]),j))
            hits=0;ap=0
            for rank,j in enumerate(order,1):
                if r[i,j]:hits+=1;ap+=hits/rank
            vals.append(ap/hits)
        return np.mean(vals)
    assert report['i2t_map_all']==pytest.approx(direct(sim,rel),abs=1e-7)
    assert report['t2i_map_all']==pytest.approx(direct(sim.T,rel.T),abs=1e-7)
    assert report['queries']==7 and report['confidence_filter'] is False


def setup_synthetic_runner(tmp_path,monkeypatch):
    v5=tmp_path/'v5';arc=v5/'diagnostics/20261107_ab150'
    (arc/'training_source').mkdir(parents=True)
    (arc/'training_source/model.py').write_text('# synthetic fixture\n')
    (v5/'data.py').write_text('# synthetic fixture\n')
    monkeypatch.setattr(m,'V5',v5)
    (v5/'runner.py').write_text(Path(m.__file__).read_text())
    monkeypatch.setattr(m,'__file__',str(v5/'runner.py'))
    monkeypatch.setattr(m,'ARCHIVE',arc)
    rng=np.random.default_rng(12);ds={}
    for split,n in [('train',13),('val',6),('test',7)]:
        ds[split]=TensorDataset(torch.tensor(rng.normal(size=(n,5)),dtype=torch.float32),
            torch.tensor(rng.normal(size=(n,6)),dtype=torch.float32),F.one_hot(torch.arange(n)%4,4).float())
    receipt=dict(image_dim=5,text_dim=6,classes=4,file_sha256='synthetic-data-fixture')
    def load(path,name):
        if path.name=='model.py':return types.SimpleNamespace(FUME=TinyFume)
        return types.SimpleNamespace(load_data=lambda *a:(ds,receipt))
    monkeypatch.setattr(m,'load_module',load)
    return ['--data',str(tmp_path/'unused.npz'),'--classes','4','--layers','1',
            '--batch-size','5','--device','cpu','--threads','1']


def test_synthetic_runner_train_resume_equal_and_no_auto_test(tmp_path,monkeypatch):
    args=setup_synthetic_runner(tmp_path,monkeypatch)
    first,second,full=(tmp_path/x for x in ['first','second','full'])
    m.main(args+['--epochs','1','--output',str(first)])
    assert not (first/'test_metrics.json').exists()
    assert json.loads((first/'history.jsonl').read_text())['train_rows']==13
    assert json.loads((first/'gradient_probe.jsonl').read_text())['output_layer_only']
    m.main(args+['--epochs','2','--output',str(second),'--resume',str(first/'last.pt')])
    m.main(args+['--epochs','2','--output',str(full),'--final-test'])
    a=torch.load(second/'last.pt',weights_only=True)
    b=torch.load(full/'last.pt',weights_only=True)
    for k in a['model']: assert torch.equal(a['model'][k],b['model'][k]),k
    assert (full/'test_metrics.json').exists()
    with pytest.raises(FileExistsError):m.main(args+['--output',str(first)])


def test_const_prediction_reference_numbers():
    p=.01
    assert -(p*math.log(p)+(1-p)*math.log(1-p))==pytest.approx(.05600153435484734)
    assert abs(math.log(p/(1-p)))*10==pytest.approx(45.9511985013459)
    assert 2*math.log(599)==pytest.approx(12.790523196230899)

from pathlib import Path
import math
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from query_mixup import query_conditioned_mixup as mix
from check_mechanisms import check

def data():
    x=torch.tensor([[1.,2.],[3.,5.],[7.,11.]],dtype=torch.float64)
    t=torch.tensor([[1.,0.,2.],[3.,1.,0.],[0.,0.,5.]],dtype=torch.float64)
    y=torch.eye(3,dtype=torch.float64)
    i=torch.tensor([0,0,1,1,2,2]);j=torch.tensor([0,1,1,2,2,0])
    p=torch.tensor([1,0,3,2,5,4])
    return x,t,y,i,j,p

@pytest.mark.parametrize('orientation',[True,False])
@pytest.mark.parametrize('coefficient',[0.,.25,.5,1.])
def test_fixed_query_and_endpoint_targets(orientation,coefficient):
    x,t,y,i,j,p=data()
    q=torch.full((6,),orientation)
    l=torch.full((6,1),coefficient,dtype=x.dtype)
    a,b,r=mix(x,t,y,i,j,permutation=p,coefficients=l,image_is_query=q)
    if orientation:
        assert torch.equal(a,x[i])
        expected=coefficient*t[j]+(1-coefficient)*t[j[p]]
        cross=(y[i]*y[j[p]]).sum(-1,keepdim=True)>0
        assert torch.equal(b,expected)
    else:
        assert torch.equal(b,t[j])
        assert torch.equal(a,coefficient*x[i]+(1-coefficient)*x[i[p]])
        cross=(y[i[p]]*y[j]).sum(-1,keepdim=True)>0
    base=(y[i]*y[j]).sum(-1,keepdim=True)>0
    assert torch.equal(r,coefficient*base.to(x.dtype)+(1-coefficient)*cross.to(x.dtype))

def test_counterexample_reproduced_and_conditioned_target_consistent():
    r=check();c=r['legacy_pair_mixup_collision'];n=r['query_conditioned_check']
    assert c['identical_images'] and c['identical_texts']
    assert c['legacy_target_from_positive_pairs']!=c['legacy_target_from_negative_pairs']
    assert n['identical_queries'] and n['identical_mixed_candidates']
    assert n['targets']==[.5,.5]

def test_soft_target_does_not_fix_strength():
    r=check()['concentration_check']
    assert all(r[i]['expected_ce']>r[i+1]['expected_ce'] for i in range(len(r)-1))
    assert all(r[i]['model_EU']>r[i+1]['model_EU'] for i in range(len(r)-1))
    assert all(x['alpha']>=1 and x['beta']>=1 for x in r)

def test_multihot_no_normalization():
    x,t,y,i,j,p=data();y[1,0]=1
    a,b,r=mix(x,t,y,i,j,permutation=p,coefficients=torch.ones(6,1,dtype=x.dtype))
    assert torch.equal(r,((y[i]*y[j]).sum(-1,keepdim=True)>0).to(x.dtype))
    assert torch.equal(y.sum(1),torch.tensor([1.,2.,1.],dtype=x.dtype))

def test_reproducible_and_gradients():
    x,t,y,i,j,p=data();x.requires_grad_();t.requires_grad_()
    torch.manual_seed(27);a=mix(x,t,y,i,j)
    torch.manual_seed(27);b=mix(x,t,y,i,j)
    assert all(torch.equal(u,v) for u,v in zip(a,b))
    (a[0].square().mean()+a[1].square().mean()).backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm()>0
    assert torch.isfinite(t.grad).all() and t.grad.norm()>0

def test_originals_unchanged_and_queries_alternate():
    x,t,y,i,j,p=data();before=(x.clone(),t.clone(),y.clone())
    a,b,r=mix(x,t,y,i,j,permutation=p)
    assert torch.equal(a[::2],x[i[::2]]) and torch.equal(b[1::2],t[j[1::2]])
    assert all(torch.equal(u,v) for u,v in zip(before,(x,t,y)))
    assert len(r)==len(i) and ((r>=0)&(r<=1)).all()

@pytest.mark.parametrize('what',['bad_beta','soft_labels','bad_perm','bad_indices','bad_lambda'])
def test_invalid_input(what):
    x,t,y,i,j,p=data();kw=dict(permutation=p)
    if what=='bad_beta':kw['beta']=-1
    if what=='soft_labels':y[0,0]=.5
    if what=='bad_perm':kw['permutation']=torch.zeros(6,dtype=torch.long)
    if what=='bad_indices':i[0]=99
    if what=='bad_lambda':kw['coefficients']=torch.full((6,1),1.5)
    with pytest.raises(ValueError):mix(x,t,y,i,j,**kw)

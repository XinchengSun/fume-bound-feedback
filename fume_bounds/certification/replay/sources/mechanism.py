"""Fixed NUS one-step comparisons. Run next to the archived real_feedback_20261110 directory."""
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import cdist
P=Path(__file__).resolve().parent
BASE=P.parent/'real_feedback_20261110'
sys.path.insert(0,str(BASE))
import experiment as e

def balanced(loss,y):
    return (loss[y].mean()+loss[~y].mean())/2

def plain_mle(feat,labels):
    # No Gaussian prior; differentiate the refitted ordinary logistic optimum.
    w=feat.new_zeros(feat.shape[1])
    def score(w):
        p=(feat@w).sigmoid()
        return feat.T@(p-labels.double()),feat.T@((p*(1-p))[:,None]*feat)
    with torch.no_grad():
        for _ in range(50):
            grad,hess=score(w)
            if grad.abs().max()<1e-9:break
            w=w-torch.linalg.solve(hess,grad)
        assert grad.abs().max()<1e-7
    grad,hess=score(w)
    return w-torch.linalg.solve(hess,grad)

def losses(wi,wt,ids):
    a,b=e.emb(e.I[ids],wi),e.emb(e.T[ids],wt)
    X=e.crossed(a,b)
    y=(e.Y[ids,None]==e.Y[ids][None,:]).flatten()
    sign=y.double()*2-1
    mu,sd=e.sufficient(X,e.state(wi,wt)).T
    pr=e.teacher(torch.stack([mu,sd],1))
    eventloss=balanced(F.softplus(-sign*mu+e.Z95*sd),y)
    torch.testing.assert_close(eventloss,e.objective(pr,y),rtol=1e-11,atol=1e-12)
    m=pr[:,2]
    map_ce=balanced(F.softplus(-sign*mu),y)
    mean_ce=balanced(torch.where(y,-m.log(),-torch.log1p(-m)),y)
    stopped=balanced(F.softplus(-sign*mu+e.Z95*sd.detach()),y)
    assert stopped.item()==eventloss.item()
    i,t=e.support
    supX=e.design(e.emb(e.I[i],wi),e.emb(e.T[t],wt))
    ordinary=balanced(F.softplus(-sign*(X@plain_mle(supX,e.Y[i]==e.Y[t]))),y)
    W=F.normalize(e.DATA['W'],dim=1)
    lab=F.one_hot(e.Y[ids],10).float()
    base=e.calc_loss(a,b,(a@W).relu(),(b@W).relu(),lab,lab,.5)
    return dict(original=base,plain=ordinary,map_logit=map_ce,predictive_mean=mean_ce,stop_sd=stopped,full=eventloss)

def map_scores(order,y):
    relevance=(y[order]==y[:,None])
    return (np.sum(relevance*(relevance.cumsum(1)/np.arange(1,len(y)+1)),axis=1)/relevance.sum(1)).mean()

@torch.no_grad()
def evaluation(wi,wt,split):
    ip,tp,y=e.DATA[split]
    a,b=e.emb(ip,wi),e.emb(tp,wt)
    ctx=e.state(wi,wt)
    dist=cdist(a.numpy(),b.numpy(),'cosine')
    ab,ba=[map_scores(v.argsort(axis=1),y.numpy()) for v in (dist,dist.T)]
    labels=(y[:,None]==y[None,:]).flatten()
    q=[]; lower=[]
    for i in range(0,len(y),32):
        mu,sd=e.sufficient(e.crossed(a[i:i+32],b),ctx).T
        q.append((mu[:,None]+sd[:,None]*e.nodes).sigmoid()@e.mass)
        sign=2*(y[i:i+32,None]==y[None,:]).flatten().double()-1
        lower.append((sign*mu-e.Z95*sd).sigmoid())
    q=torch.cat(q);lower=torch.cat(lower)
    bins=(10*q).long().clamp(max=9)
    return dict(i2t=ab,t2i=ba,map=(ab+ba)/2,nll=F.binary_cross_entropy(q,labels.double()).item(),
                brier=(q-labels.double()).square().mean().item(),ece=torch.bincount(bins,
                weights=q-labels.double(),minlength=10).abs().sum().item()/len(q),
                pos_L=lower[labels].mean().item(),neg_L=lower[~labels].mean().item(),
                event_loss=balanced(-lower.log(),labels).item())

def main():
    g=torch.Generator().manual_seed(110104)
    batches=[e.batch,*e.perm[torch.randperm(5600,generator=g)[:2100]].reshape(7,300)]
    assert len(torch.cat(batches).unique())==2400
    assert not np.intersect1d(torch.cat(batches),e.support.flatten()).size
    (P/'batches.json').write_text(json.dumps([v.tolist() for v in batches]))
    historic=json.loads((BASE/'result.json').read_text())
    wi,wt=[e.DATA[k].detach().requires_grad_() for k in ('wI','wT')]
    rows=[];checks=[]
    def record(batch,group,kind,weights,objective=None):
        for split in ('val','test'):
            data=evaluation(*weights,split)
            if batch==0 and (kind=='original' or group=='frozen'):
                key='frozen' if group=='frozen' else 'original'
                assert abs(data['map']-historic['arms'][key][split]['map'])<1e-11
            rows.append(dict(batch=batch,group=group,kind=kind,split=split,loss=objective,**data))
    record(0,'frozen','frozen',(wi,wt))
    for j,ids in enumerate(batches):
        terms=losses(wi,wt,ids)
        grads={k:e.gradients(v,wi,wt) for k,v in terms.items()}
        for i,name in enumerate(('image','text')):
            full,stop,point=[grads[k][i].flatten() for k in ('full','stop_sd','map_logit')]
            checks.append(dict(batch=j,tower=name,full_stop_cos=F.cosine_similarity(full,stop,dim=0).item(),
                               sd_relative_norm=((full-stop).norm()/full.norm()).item(),
                               weighting_relative_norm=((stop-point).norm()/point.norm()).item(),
                               full_stop_sign_disagree=(full.sign()!=stop.sign()).float().mean().item()))
        for kind,grad in grads.items():
            for group in (['base'] if kind=='original' else ['alone','plus_FUME']):
                w=[z.detach().clone() for z in (wi,wt)]
                opt=torch.optim.Adam(w,lr=5e-5,betas=(.5,.999))
                for i,v in enumerate(w):
                    v.grad=grad[i]+grads['original'][i] if group=='plus_FUME' else grad[i]
                opt.step()
                record(j,group,kind,w,terms[kind].item()+(terms['original'].item() if group=='plus_FUME' else 0))
        for path,values in [('raw.csv',rows),('gradients.csv',checks)]:
            with (P/path).open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=values[0]);w.writeheader();w.writerows(values)
        print('completed batch',j,flush=True)
if __name__=='__main__':main()


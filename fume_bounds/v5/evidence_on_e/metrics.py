"""Auxiliary Beta uncertainty metrics; not a replacement for embedding cosine retrieval."""
import numpy as np
import torch


def statistics(a, b):
    a, b = a.double(), b.double()
    s = a+b
    p = a/s
    total = -torch.xlogy(p,p)-torch.xlogy(1-p,1-p)
    aleatoric = torch.digamma(s+1)-p*torch.digamma(a+1)-(1-p)*torch.digamma(b+1)
    return dict(probability=p,total_entropy=total,aleatoric_entropy=aleatoric,
                epistemic_mi=(total-aleatoric).clamp_min(0),vacuity=2/s)


def error_auroc(error, score):
    """Mann-Whitney AUROC with average ranks for exact ties; None for one class."""
    y,s=np.asarray(error).ravel().astype(bool),np.asarray(score).ravel()
    pos,neg=int(y.sum()),int((~y).sum())
    if not pos or not neg:
        return None
    order=np.argsort(s,kind='stable')
    ss=s[order]
    starts=np.r_[0,np.flatnonzero(ss[1:]!=ss[:-1])+1]
    ends=np.r_[starts[1:],len(y)]
    ranks=np.repeat((starts+1+ends)/2,ends-starts)
    return float((ranks[y[order]].sum()-pos*(pos+1)/2)/(pos*neg))


@torch.no_grad()
def evidence_report(model,dataset,pairs=20000,seed=20261107,chunk=1024):
    """Fixed sampled pairs or labels. Balanced pairs differ from gallery's class prior."""
    if model.task=='none':
        return dict(task='none',reason='No evidence head')
    if min(pairs,chunk)<1:
        raise ValueError('Positive evaluation sizes required')
    state=torch.get_rng_state()
    old=model.training
    device=next(model.parameters()).device
    model.eval()
    try:
        xi,xt,y=dataset.tensors
        zs={'image':[],'text':[]}
        params={'image':[],'text':[]}
        for start in range(0,len(y),128):
            views=model.encode(xi[start:start+128].to(device),xt[start:start+128].to(device))
            for side,row in views.items():
                zs[side].append(row['z'].cpu())
                if model.task=='category':
                    o=model.category(row,side)
                    params[side].append((o['alpha'].cpu(),o['beta'].cpu()))
        if model.task=='matching':
            gen=torch.Generator().manual_seed(seed)
            overlap=(y.float()@y.float().T)>0
            n=len(y)
            anchors=torch.randint(n,(max(1,pairs//2),),generator=gen)
            pos=torch.multinomial(overlap[anchors].float(),1,generator=gen).squeeze(1)
            valid=(~overlap[anchors]).any(1)
            an=anchors[valid]
            neg=torch.multinomial((~overlap[an]).float(),1,generator=gen).squeeze(1) if len(an) else an
            ii=torch.cat((anchors,an))
            jj=torch.cat((pos,neg))
            target=torch.cat((torch.ones(len(anchors),1),torch.zeros(len(an),1)))
            image,text=(torch.cat(zs[v]) for v in ('image','text'))
            vals=[]
            for start in range(0,len(ii),chunk):
                o=model.pair_head(image[ii[start:start+chunk]].to(device),
                                  text[jj[start:start+chunk]].to(device))
                vals.append((o['alpha'].cpu(),o['beta'].cpu()))
            params={'matching':vals}
            targets={'matching':target}
        else:
            targets={'image':y,'text':y}
        report=dict(task=model.task,model_implied_uncertainty=True,
                    event='sampled image-text relevance' if model.task=='matching' else 'label presence',
                    seed=seed,balanced_pair_sampling=model.task=='matching',views={})
        for side,values in params.items():
            a,b=(torch.cat([v[k] for v in values]) for k in (0,1))
            t=targets[side].double()
            stats=statistics(a,b)
            p=stats['probability']
            error=(p>=.5)!=t.bool()
            info=dict(decisions=t.numel(),positive_rate=float(t.mean()),
                      brier=float((p-t).square().mean()),
                      nll=float(-(t*p.log()+(1-t)*torch.log1p(-p)).mean()),
                      mean_concentration=float((a+b).mean()))
            for key in ('epistemic_mi','aleatoric_entropy','vacuity'):
                info[key]=dict(mean=float(stats[key].mean()),error_auroc=error_auroc(error,stats[key]))
            report['views'][side]=info
        return report
    finally:
        model.train(old)
        torch.set_rng_state(state)

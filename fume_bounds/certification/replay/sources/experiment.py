"""NUS single-step audit. Four-output student on a support-conditioned logistic-normal reference."""
from pathlib import Path
import json,sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from numpy.polynomial.hermite import hermgauss
from scipy.spatial.distance import cdist

P=Path(__file__).resolve().parent
OFF=P.parents[1]/'2025-CVPR-FUME'
sys.path.insert(0,str(OFF))
from evaluate import fx_calc_map_label
from processor import calc_loss
torch.set_num_threads(3)
torch.manual_seed(41531)
rng=np.random.default_rng(41531)
roots,weights=hermgauss(24)
nodes=torch.tensor(roots*np.sqrt(2))
mass=torch.tensor(weights/np.sqrt(np.pi))
Z95=1.6448536269514722
DATA=torch.load(P/'features.pt',map_location='cpu',weights_only=True)
I,T,Y=DATA['train']
perm=torch.tensor(rng.permutation(len(Y)))
support=perm[5600:7648].reshape(2,-1)
batch=perm[-300:]
def emb(x,w):return F.normalize(F.linear(x,w),dim=1)
def design(a,b):
    v=(a*b).double()*10
    return torch.cat([torch.ones_like(v[:,:1]),v],dim=1)

def posterior(X,y):
    # Independent N(0,4) priors. The extra Newton step differentiates the MAP
    # implicitly at stationarity even though the preliminary solve is detached.
    y=y.double()
    prior=torch.eye(X.shape[1],dtype=X.dtype)/4
    def derivatives(w,D):
        p=(D@w).sigmoid()
        h=D.T@((p*(1-p))[:,None]*D)+prior
        return h,D.T@(p-y)+prior@w
    with torch.no_grad():
        w=X.new_zeros(X.shape[1])
        for _ in range(40):
            h,g=derivatives(w,X)
            if g.abs().max()<1e-9:break
            w=w-torch.linalg.solve(h,g)
        assert g.abs().max()<1e-7
    h,g=derivatives(w,X)
    w=w-torch.linalg.solve(h,g)
    h,_=derivatives(w,X)
    return w,torch.linalg.inv(h)

def state(wi,wt,ids=support):
    a,b=ids
    return posterior(design(emb(I[a],wi),emb(T[b],wt)),Y[a]==Y[b])

def sufficient(X,context):
    w,cov=context
    return torch.stack([X@w,((X@cov)*X).sum(1).sqrt()],1)
def entropy_from_logit(t):
    return F.softplus(-t.abs())+t.abs()*(-t.abs()).sigmoid()
def entropy(p):
    return -torch.xlogy(p,p)-torch.xlogy(1-p,1-p)
def teacher(inp):
    mu,sd=inp.T
    samples=mu[:,None]+sd[:,None]*nodes
    m=samples.sigmoid()@mass
    au=entropy_from_logit(samples)@mass
    return torch.stack([(mu-Z95*sd).sigmoid(),(mu+Z95*sd).sigmoid(),m,au,entropy(m)-au],1)

class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),nn.Linear(48,4))
    def forward(self,inp):
        mu,sd=inp.T
        x=torch.stack([mu/4,sd.log()/2],1)
        left,span,center,part=self.net(x.float()).double().T
        l=(mu+sd*left).sigmoid()
        u=(mu+sd*(left+F.softplus(span))).sigmoid()
        t=mu+sd*center
        h=entropy_from_logit(t)
        au=h*(-part-2*sd.log()).sigmoid()
        return torch.stack([l,u,t.sigmoid(),au,h-au],1)

def distill(wi,wt):
    allx=[]
    a,b=emb(I,wi),emb(T,wt)
    for i in range(36):
        ids=perm[torch.tensor(rng.permutation(3500)[:2*(512,1024,1536)[i%3]])].reshape(2,-1)
        context=posterior(design(a[ids[0]],b[ids[1]]),Y[ids[0]]==Y[ids[1]])
        ix=perm[torch.tensor(rng.integers(3500,5600,(2,3072)))]
        allx.append(sufficient(design(a[ix[0]],b[ix[1]]),context))
    x=torch.cat(allx);y=teacher(x)
    train_n=30*3072
    scales=y[:train_n].mean(0)
    model=Student()
    opt=torch.optim.Adam(model.parameters(),lr=.002)
    trace=[];best=float('inf')
    for ep in range(25):
        loss_sum=0.
        for ix in torch.randperm(train_n).split(1024):
            loss=((model(x[ix])-y[ix]).abs()/scales).mean()
            opt.zero_grad();loss.backward();opt.step()
            loss_sum+=loss.item()*len(ix)
        with torch.no_grad():
            val=((model(x[train_n:])-y[train_n:]).abs()/scales).mean().item()
        trace.append([ep+1,loss_sum/train_n,val])
        if val<best:
            best=val
            torch.save(model.state_dict(),P/'student.pt')
    np.savetxt(P/'distill_trace.csv',trace,delimiter=',',header='epoch,train,val',comments='')
    model.load_state_dict(torch.load(P/'student.pt',weights_only=True))
    return model.eval().requires_grad_(False)

def crossed(a,b):
    return design(a[:,None,:].expand(-1,len(b),-1).reshape(-1,10),b.repeat(len(a),1))
def event(out,rel):
    return torch.where(rel,out[:,0],1-out[:,1])
def objective(out,rel):
    bounds=event(out,rel)
    return -.5*(bounds[rel].log().mean()+bounds[~rel].log().mean())

def batch_terms(wi,wt,student):
    a,b=emb(I[batch],wi),emb(T[batch],wt)
    c=state(wi,wt)
    args=sufficient(crossed(a,b),c)
    rel=(Y[batch,None]==Y[batch][None,:]).flatten()
    st,src=student(args),teacher(args)
    W=F.normalize(DATA['W'],dim=1)
    lbl=F.one_hot(Y[batch],10).float()
    base=calc_loss(a,b,(a@W).relu(),(b@W).relu(),lbl,lbl,.5)
    return base,objective(st,rel),objective(src,rel),[float(v.detach()) for v in
        (st[rel,0].mean(),src[rel,0].mean(),(1-st[~rel,1]).mean(),(1-src[~rel,1]).mean())]

def gradients(loss,wi,wt):return torch.autograd.grad(loss,(wi,wt),retain_graph=True)

@torch.no_grad()
def evaluate(wi,wt,split,student):
    ip,tp,y=DATA[split]
    a,b=emb(ip,wi),emb(tp,wt)
    context=state(wi,wt)
    maps=[float(fx_calc_map_label(u.numpy(),v.numpy(),y.numpy(),0)) for u,v in [(a,b),(b,a)]]
    labels=(y[:,None]==y[None,:]).reshape(-1)
    chunks=[[],[]]
    for part in a.split(16):
        x=sufficient(crossed(part,b),context)
        for rows,val in zip(chunks,[student(x),teacher(x)]):rows.append(val)
    st,src=[torch.cat(v) for v in chunks]
    def metrics(o):
        q=o[:,2]
        prob=q.clamp(1e-15,1-1e-15)
        bins=(q*10).long().clamp(max=9)
        gap=torch.bincount(bins,weights=q-labels.double(),minlength=10).abs().sum()/len(q)
        return dict(pos_L=o[labels,0].mean().item(),neg_L=(1-o[~labels,1]).mean().item(),
                    feedback_loss=objective(o,labels).item(),brier=(q-labels.double()).square().mean().item(),
                    nll=F.binary_cross_entropy(prob,labels.double()).item(),ece10=gap.item())
    return dict(i2t=maps[0],t2i=maps[1],map=np.mean(maps),pairs=len(labels),source=metrics(src),
        student=metrics(st),mae=(st-src).abs().mean(0).tolist(),bias=(st-src).mean(0).tolist())

def main():
    wi,wt=DATA['wI'],DATA['wT']
    student=distill(wi,wt)
    wi,wt=[w.detach().requires_grad_() for w in (wi,wt)]
    base,fb,src,parts=batch_terms(wi,wt,student)
    gf,gs=gradients(fb,wi,wt),gradients(src,wi,wt)
    audit=dict(gradient_cos=[F.cosine_similarity(a.flatten(),b.flatten(),dim=0).item() for a,b in zip(gf,gs)],
               student_grad_norm=[x.norm().item() for x in gf],source_grad_norm=[x.norm().item() for x in gs])
    out={}
    for arm,loss in [('frozen',None),('original',base),('feedback',fb),('combined',base+fb)]:
        w=[v.detach().clone() for v in (wi,wt)]
        if loss is not None:
            g=gradients(loss,wi,wt)
            # Fresh Adam, same step and hyperparameters for all update arms.
            opt=torch.optim.Adam(w,lr=5e-5,betas=(.5,.999))
            for v,grad in zip(w,g):v.grad=grad
            opt.step()
        with torch.no_grad():
            bl,fl,sl,p=batch_terms(*w,student)
            out[arm]=dict(batch=dict(base=bl.item(),student_loss=fl.item(),source_loss=sl.item(),
                student_pos=p[0],source_pos=p[1],student_neg=p[2],source_neg=p[3]))
            for split in ('val','test'):
                out[arm][split]=evaluate(*w,split,student)
        if arm=='feedback':
            delta=[v-old.detach() for v,old in zip(w,(wi,wt))]
            audit['slope']={}
            for label,gr,j in [('student',gf,1),('source',gs,2)]:
                # .05 of the actual update on either side, with full support refit.
                with torch.no_grad():
                    l=[batch_terms(*[v+h*d for v,d in zip((wi,wt),delta)],student)[j].item() for h in (-.05,.05)]
                audit['slope'][label]=[sum((u*d).double().sum().item() for u,d in zip(gr,delta)),(l[1]-l[0])/.1]
        (P/'result.json').write_text(json.dumps(dict(gradient=audit,arms=out),indent=2,allow_nan=False))
        print(arm,out[arm]['test']['map'],flush=True)

if __name__=='__main__':main()


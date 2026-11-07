"""E membership supervision with exactly one optional category or matching evidence head.

Clean loss calls the independently tested E minimal objective, never BCE, fuzzy c or necessity.
Auxiliary evidence uses the previous v5 parameters and Softplus+1 prior.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
V5 = HERE.parent
BASE_PATH = V5 / 'minimal_supervision/minimal_control.py'
_spec = importlib.util.spec_from_file_location('_e_evidence_baseline', BASE_PATH)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
LEGACY_PATH = base.ARCHIVE / 'training_source/model.py'


@dataclass(frozen=True)
class AuxConfig:
    vicinal: float = 1.0
    noise: float = 1.0
    error_kl: float = 0.01
    kl_anneal_epochs: int = 10
    mix_beta: float = 10.0
    noise_beta: float = 1.0
    noise_multiplier: float = 1.0
    pair_negatives: int = 1
    pair_positive_mode: str = 'overlap'
    pair_chunk: int = 128
    pair_clean_weight: float = 0.0

    def validate(self):
        for name in ('vicinal', 'noise', 'error_kl', 'mix_beta', 'noise_beta',
                     'noise_multiplier', 'pair_clean_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'Invalid {name}')
        if min(self.mix_beta, self.noise_beta) <= 0:
            raise ValueError('Beta sampling parameters must be positive')
        for name in ('kl_anneal_epochs', 'pair_negatives', 'pair_chunk'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'Invalid {name}')
        if self.pair_chunk < 1 or self.pair_positive_mode not in ('overlap', 'aligned'):
            raise ValueError('Invalid pair chunk/positive mode')

    def kl_weight(self, epoch):
        if type(epoch) is not int or epoch < 1:
            raise ValueError('Epoch is 1-based')
        return self.error_kl * (1.0 if self.kl_anneal_epochs <= 1 else
                               min(1.0, (epoch-1)/(self.kl_anneal_epochs-1)))


class PairHead(nn.Module):
    """4d+1 -> 128 -> 2 / ReLU / Softplus+1, same as old v5 pair evidence."""
    def __init__(self, dimension, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4*dimension+1, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, image, text):
        product = image*text
        logits = self.net(torch.cat((image, text, (image-text).abs(), product,
                                    product.sum(-1, keepdim=True)), -1))
        params = 1 + F.softplus(logits)
        return {'alpha': params[:, :1], 'beta': params[:, 1:]}


class EvidenceOnE(nn.Module):
    def __init__(self, image_dim, text_dim, classes, layers=4, task='matching',
                 initial_evidence=2.0, pair_hidden=128):
        super().__init__()
        if task not in ('none', 'category', 'matching'):
            raise ValueError('task must be none/category/matching')
        if any(type(v) is not int or v < 1 for v in (image_dim, text_dim, classes, layers, pair_hidden)):
            raise ValueError('Positive integer dimensions required')
        if classes < 2 or not math.isfinite(initial_evidence) or initial_evidence <= 0:
            raise ValueError('Invalid classes/initial evidence')
        legacy = base.load_module(LEGACY_PATH, '_e_evidence_original_fume')
        original = legacy.FUME('cpu', image_dim, text_dim, classes, classes, layers)
        self.img_net, self.text_net = original.img_net, original.text_net
        self.register_buffer('W', original.W)
        self.task = task
        self.config = dict(image_dim=image_dim, text_dim=text_dim, classes=classes,
                           layers=layers, task=task, initial_evidence=initial_evidence,
                           pair_hidden=pair_hidden)
        with torch.random.fork_rng(devices=[]):
            if task == 'matching':
                self.pair_head = PairHead(classes, pair_hidden)
            elif task == 'category':
                self.img_evidence = nn.Linear(4096, 1)
                self.text_evidence = nn.Linear(4096, 1)
                bias = initial_evidence + math.log(-math.expm1(-initial_evidence))
                for head in (self.img_evidence, self.text_evidence):
                    nn.init.zeros_(head.weight)
                    nn.init.constant_(head.bias, bias)

    @contextmanager
    def capture(self):
        """Read tensors from the original FUME forward without replacing the original network."""
        views, handles = {}, []
        for name, tower in (('image', self.img_net), ('text', self.text_net)):
            row = views[name] = {}
            def linear_hook(module, args, result, row=row):
                row['hidden'], row['h'] = args[0], result
            def tower_hook(module, args, result, row=row):
                row['z'] = result
            handles.extend([tower.layers[-1].register_forward_hook(linear_hook),
                            tower.register_forward_hook(tower_hook)])
        try:
            yield views
        finally:
            for handle in handles:
                handle.remove()

    def encode(self, image, text):
        with self.capture() as views:
            self.img_net(image)
            self.text_net(text)
        return views

    def category(self, row, modality):
        if self.task != 'category':
            raise ValueError('No category evidence head')
        head = self.img_evidence if modality == 'image' else self.text_evidence
        direction = (row['h'] @ self.W).sigmoid()
        s = F.softplus(head(row['hidden']))
        return dict(alpha=1+s*direction, beta=1+s*(1-direction), evidence_total=s)

    def matching(self, image, text):
        if self.task != 'matching':
            raise ValueError('No matching head')
        return self.pair_head(self.img_net(image), self.text_net(text))


def expected_ce(out, target):
    a, b, t = out['alpha'].double(), out['beta'].double(), target.double()
    if a.shape != t.shape or b.shape != a.shape or not torch.isfinite(t).all() or (t<0).any() or (t>1).any():
        raise ValueError('Invalid evidence targets/shapes')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or (a<1).any() or (b<1).any():
        raise ValueError('Finite evidence+1 parameters required')
    return (t*(torch.digamma(a+b)-torch.digamma(a))+
            (1-t)*(torch.digamma(a+b)-torch.digamma(b))).mean()


def wrong_kl(out, y):
    if not ((y==0)|(y==1)).all():
        raise ValueError('KL is for hard labels ONLY')
    if out['alpha'].shape != y.shape or out['beta'].shape != y.shape:
        raise ValueError('KL shape mismatch')
    q = torch.where(y.bool(), out['beta'], out['alpha']).double()
    if not torch.isfinite(q).all() or (q<1).any():
        raise ValueError('KL requires evidence+1')
    return (torch.log1p(q-1)-(q-1)/q).mean()


def sample_pairs(y, cfg):
    """Sample label-overlap positives and disjoint-label negatives; do not drop clean samples."""
    cfg.validate()
    if y.ndim != 2 or not ((y==0)|(y==1)).all() or (y.sum(1)==0).any():
        raise ValueError('Nonempty hard labels required')
    overlap = (y.float() @ y.float().T)>0
    n, device = len(y), y.device
    anchor = torch.arange(n, device=device)
    pos = anchor if cfg.pair_positive_mode == 'aligned' else torch.multinomial(overlap.float(),1).squeeze(1)
    rev = torch.rand(n, device=device)<.5
    ii, jj = [torch.where(rev,pos,anchor)], [torch.where(rev,anchor,pos)]
    targets = [torch.ones(n,1,device=device)]
    available = (~overlap).any(1)
    candidates = (~overlap).float()
    candidates[~available] = 0
    candidates[~available,anchor[~available]] = 1
    for _ in range(cfg.pair_negatives):
        neg = torch.multinomial(candidates,1).squeeze(1)
        rev = torch.rand(n,device=device)<.5
        ii.append(torch.where(rev,neg,anchor)[available])
        jj.append(torch.where(rev,anchor,neg)[available])
        targets.append(torch.zeros(int(available.sum()),1,device=device))
    i,j,t = torch.cat(ii),torch.cat(jj),torch.cat(targets)
    if not torch.equal(overlap[i,j],t[:,0].bool()):
        raise RuntimeError('Invalid overlap supervision')
    return i,j,t,dict(pair_count=len(i),pair_positives=n,pair_negatives=len(i)-n,
                      anchors_without_negative=int((~available).sum()))


def augment(image, text, y, cfg, scales, kind):
    if kind != 'noise':
        raise ValueError('Use mixup() for vicinal inputs')
    lam = torch.distributions.Beta(cfg.noise_beta,cfg.noise_beta).sample((len(y),1)).to(image)
    return (lam*image+(1-lam)*torch.randn_like(image)*scales[0]*cfg.noise_multiplier,
            lam*text+(1-lam)*torch.randn_like(text)*scales[1]*cfg.noise_multiplier,
            lam*y+(1-lam)*.5)


def mixup(image, text, y, cfg):
    p = torch.randperm(len(y),device=y.device)
    lam = torch.distributions.Beta(cfg.mix_beta,cfg.mix_beta).sample((len(y),1)).to(image)
    return lam*image+(1-lam)*image[p], lam*text+(1-lam)*text[p], lam*y+(1-lam)*y[p]


def objective(model, batch, cfg=AuxConfig(), scales=(1.,1.), *, epoch=1, backward=False, probe=False, cl_weight=0.05):
    """Accumulate all branches; caller performs exactly one optimizer step per original batch."""
    cfg.validate()
    if not math.isfinite(float(cl_weight)) or cl_weight < 0:
        raise ValueError('Contrastive weight must be finite and nonnegative')
    if len(scales)!=2 or any(not math.isfinite(float(s)) or s<0 for s in scales):
        raise ValueError('Two finite nonnegative training noise scales required')
    if model.task!='matching' and cfg.pair_clean_weight:
        raise ValueError('pair_clean_weight is matching-only')
    xi,xt,y = batch
    terms, logs, grads = [], {}, {}
    weights = (model.img_net.layers[-1].weight, model.text_net.layers[-1].weight)
    def observe(name, value, weight=1.):
        if not probe or not weight:
            return
        gs = torch.autograd.grad(weight*value,weights,retain_graph=True,allow_unused=True)
        gs = [torch.zeros_like(w) if g is None else g.detach() for g,w in zip(gs,weights)]
        if name in grads:
            gs = [a+b for a,b in zip(grads[name],gs)]
        grads[name] = gs
    def finish(name, value, weight=1.):
        if not torch.isfinite(value):
            raise FloatingPointError(f'Nonfinite {name}')
        logs[name] = logs.get(name,0.)+float(value.detach())
        logs['weighted_'+name] = logs.get('weighted_'+name,0.)+weight*float(value.detach())
        part = weight*value
        if backward:
            part.backward()
            terms.append(part.detach())
        else:
            terms.append(part)

    with model.capture() as views:
        _original_e_total,sup,cl = base.minimal_objective(model,xi,xt,y)
    # Preserve E's exact membership term. Only the explicitly configured
    # contrastive coefficient may differ from the historical E coefficient .05.
    clean = sup + cl_weight*cl
    del _original_e_total
    logs.update(supervision=float(sup.detach()),contrastive=float(cl.detach()),
                weighted_contrastive=cl_weight*float(cl.detach()),
                contrastive_weight=float(cl_weight),kl_coefficient=cfg.kl_weight(epoch))
    observe('supervision',sup)
    observe('contrastive',cl,cl_weight)
    for name,row in views.items():
        logs['h_norm_'+name] = float(row['h'].detach().norm(dim=1).mean())
    active = model.task!='none' and (cfg.vicinal or cfg.noise or cfg.error_kl or cfg.pair_clean_weight)
    if active and model.task=='matching':
        i,j,target,counts = sample_pairs(y,cfg)
        logs.update(counts)
    if active and model.task=='category':
        outputs = {v:model.category(row,v) for v,row in views.items()}
        for v,out in outputs.items():
            logs['evidence_'+v] = float(out['evidence_total'].detach().mean())
    kw = cfg.kl_weight(epoch) if active else 0.
    if active and (kw or cfg.pair_clean_weight):
        if model.task=='matching':
            out = model.pair_head(views['image']['z'][i],views['text']['z'][j])
            kl = wrong_kl(out,target) if kw else None
            if cfg.pair_clean_weight:
                pcl=expected_ce(out,target)
                observe('pair_clean',pcl,cfg.pair_clean_weight)
                logs['pair_clean']=float(pcl.detach())
                logs['weighted_pair_clean']=cfg.pair_clean_weight*logs['pair_clean']
                clean=clean+cfg.pair_clean_weight*pcl
            del out
        else:
            kl=(wrong_kl(outputs['image'],y)+wrong_kl(outputs['text'],y))/2
        if kw:
            observe('error_kl',kl,kw)
            logs['error_kl']=float(kl.detach())
            logs['weighted_error_kl']=kw*logs['error_kl']
            clean=clean+kw*kl
            del kl
    finish('clean_total',clean)
    del clean,sup,cl,views
    if active and model.task=='category':
        del outputs
    if active:
        inputs=(xi[i],xt[j],target) if model.task=='matching' else (xi,xt,y)
        for name,weight in (('vicinal',cfg.vicinal),('noise',cfg.noise)):
            if not weight:
                continue
            data=mixup(*inputs,cfg) if name=='vicinal' else augment(*inputs,cfg,scales,'noise')
            if model.task=='matching':
                m=len(data[2])
                for start in range(0,m,cfg.pair_chunk):
                    end=min(start+cfg.pair_chunk,m)
                    out=model.matching(data[0][start:end],data[1][start:end])
                    value=expected_ce(out,data[2][start:end])*((end-start)/m)
                    observe(name,value,weight)
                    finish(name,value,weight)
                    del out,value
            else:
                rows=model.encode(data[0],data[1])
                value=sum(expected_ce(model.category(rows[v],v),data[2]) for v in rows)/2
                observe(name,value,weight)
                finish(name,value,weight)
                del rows,value
            del data
    total=sum(terms)
    logs['total']=float(total.detach())
    summary={}
    if probe:
        for index,side in enumerate(('image','text')):
            summary[side]={}
            reference=grads['supervision'][index].double()
            for name,gs in grads.items():
                g=gs[index].double()
                denom=reference.norm()*g.norm()
                summary[side][name]=dict(weighted_grad_norm=float(g.norm()),
                    cosine_to_supervision=float((reference*g).sum()/denom) if denom>0 else None)
    return total,logs,summary

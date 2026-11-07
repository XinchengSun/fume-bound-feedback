"""Independent INRIA audit. Read-only model/data inputs. All forwards take place on CPU."""
import sys, os, json, math, gc
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn import functional as F

own=Path('/inspire/ssd/project/video-generation/public/sunxincheng/FUME')
base=own/'local_ops/v5_20261106'
v5=base/'repo/fume_bounds/v5'
run=own/'runs/inria_ab150_seed2_20261107'
sys.path.insert(0,str(base/'validation'))
import compare150 as runlib
from run_evidence_ab import build_model,ModelConfig
from losses import fume_contrastive
torch.set_num_threads(2)
np.random.seed(864)
torch.manual_seed(864)

def calc_map(x,t,ids,details=True):
    # Independent SciPy float64 cosine; integer relevance and AP at the actual positive ranks.
    sims=1-cdist(x,t,metric='cosine')
    labels=ids[:,None]==ids
    assert np.isfinite(sims).all()
    out={}
    for direction,z in [('i2t',sims),('t2i',sims.T)]:
        ranks=np.argsort(-z,axis=1,kind='stable')
        aps=[]
        for y,row in zip(ids,ranks):
            hits=np.flatnonzero(ids[row]==y)+1
            aps.append(np.mean(np.arange(1,len(hits)+1)/hits))
        out[direction]=float(np.mean(aps))
    out['mean']=(out['i2t']+out['t2i'])/2
    if details:
        same=labels.copy()
        np.fill_diagonal(same,False)
        out.update(paired_cos=float(sims.diagonal().mean()),same_class_nonpaired_cos=float(sims[same].mean()),
                   different_class_cos=float(sims[~labels].mean()),cos_sd=float(sims.std()))
        n=len(ids)
        h=sum(1/i for i in range(1,n+1))
        row_counts=labels.sum(1)
        out['uniform_random_expected_map']=float(h/n+np.mean(row_counts-1)/(n-1)*(1-h/n))
    return out

def snapshot(arm,data):
    folder=run/arm
    ck=torch.load(folder/'last.pt',weights_only=True,map_location='cpu',mmap=True)
    model=build_model(ModelConfig(),'matching' if arm=='D_matching' else 'category')
    if arm=='A_fume':
        weight=ck['model']
        model.img_net.load_state_dict({k.removeprefix('img_net.'):v for k,v in weight.items() if k.startswith('img_net.')})
        model.text_net.load_state_dict({k.removeprefix('text_net.'):v for k,v in weight.items() if k.startswith('text_net.')})
        model.W.copy_(weight['W'])
    else:
        model.load_state_dict(ck['model_state'])
    model.eval()
    result={'epoch':ck['epoch']}
    del ck
    for split,ds in [('val',data['val']),('train_fixed_subset',Subset(data['train'],np.random.default_rng(864).choice(len(data['train']),len(data['val']),replace=False).tolist()))]:
        parts=[[] for _ in range(5)]
        with torch.inference_mode():
            for xi,xt,y in DataLoader(ds,batch_size=128):
                vi,vt=model(xi,xt)
                for stack,var in zip(parts,[vi['z'],vt['z'],vi['logits'].sigmoid(),vt['logits'].sigmoid(),y]):
                    stack.append(var.numpy())
        xi,xt,pi,pt,labels=map(np.concatenate,parts)
        assert np.all(labels.sum(1)==1)
        ids=labels.argmax(1)
        metrics={'embedding':calc_map(xi,xt,ids)}
        if arm!='A_fume':
            metrics['class_probability_cosine_diagnostic']=calc_map(pi,pt,ids,False)
            for name,p,x in [('image',pi,xi),('text',pt,xt)]:
                metrics[name]={'classification_top1':float(np.mean(p.argmax(1)==ids)),
                   'norm_of_mean_embedding':float(np.linalg.norm(x.mean(0))),
                   'top_singular_energy_fraction':float(np.linalg.svd(x,compute_uv=False)[0]**2/len(x))}
        result[split]=metrics
    valrows=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
    result['logged_val_mean']=next(row['mean_map'] for row in valrows if row['epoch']==result['epoch'])
    return result

def check_cl():
    xi=F.normalize(torch.randn(83,100,dtype=torch.float64),dim=1).requires_grad_()
    xt=F.normalize(torch.randn(83,100,dtype=torch.float64),dim=1).requires_grad_()
    y=F.one_hot(torch.arange(83)%19,100).double()
    pred1=xi.detach().relu(); pred2=xt.detach().relu()
    original=runlib.legacy_loss_fn()
    args=(xi,xt,pred1,pred2,y,y)
    old=original(*args,1.)-original(*args,0.)
    new=fume_contrastive(xi,xt)
    g1=torch.autograd.grad(old,(xi,xt)); g2=torch.autograd.grad(new,(xi,xt))
    return {'value_diff':abs(float((old-new).detach())), 'max_abs_grad_diff':max(float((a-b).abs().max()) for a,b in zip(g1,g2))}

def main():
    data,receipt=runlib.checked_data()
    out={'data_sha':receipt['file_sha256'],'cl_check':check_cl(),'arms':{}}
    original_orders=(run/'A_fume/order.jsonl').read_text().splitlines()
    arms=['A_fume','B_bce','C_category']
    if (run/'D_matching/last.pt').exists():
        arms.append('D_matching')
    for arm in arms:
        row=snapshot(arm,data)
        rows=(run/arm/'order.jsonl').read_text().splitlines()[:row['epoch']]
        row['orders_match']=rows==original_orders[:row['epoch']]
        if arm!='A_fume' and (run/arm/'test/metrics.json').is_file():
            p=run/arm/'test'
            with np.load(p/'image_common_predictions.npz') as i,np.load(p/'text_common_predictions.npz') as t:
                y=data['test'].tensors[-1].numpy()
                assert np.array_equal(i['labels'],y) and np.array_equal(t['labels'],y)
                row['test_independent']=calc_map(i['z'],t['z'],y.argmax(1))
            row['test_official']=json.loads((p/'metrics.json').read_text())['retrieval']
        out['arms'][arm]=row
        print(arm,row['epoch'],row['val']['embedding']['mean'],flush=True,file=sys.stderr)
        gc.collect()
    Path(sys.argv[1]).write_text(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':
    main()

"""One synthetic CPU Adam update at FULL default FUME dimensions; not a benchmark."""
import json
from pathlib import Path
import time
import torch
from model import ModelConfig
from evidence_ab import build_model, PairConfig, ab_objective
from experiment import ExperimentLoss

def main():
    torch.set_num_threads(1);torch.manual_seed(2)
    m=build_model(ModelConfig(),'matching')
    x,t=torch.randn(2,4096),torch.randn(2,1000)
    y=torch.zeros(2,100);y[0,0]=1;y[0,2]=1;y[1,1]=1
    names=('img_net.layers.4.weight','text_net.layers.4.weight','pair_head.net.2.weight')
    probes={n:dict(m.named_parameters())[n].detach().clone() for n in names}
    opt=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.5,.999))
    clock=time.perf_counter()
    loss,logs=ab_objective(m,(x,t,y),ExperimentLoss(error_kl=.01),(1.,1.),epoch=2,
                          backward=True,pair_config=PairConfig(chunk=2))
    grads={n:float(dict(m.named_parameters())[n].grad.norm()) for n in names}
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    opt.step()
    changes={n:float((dict(m.named_parameters())[n].detach()-probes[n]).abs().max()) for n in names}
    assert all(v>0 for v in grads.values()) and all(v>0 for v in changes.values())
    report=dict(torch=str(torch.__version__),device='cpu',data='synthetic batch2, not INRIA',
                parameters=sum(p.numel() for p in m.parameters()),loss=float(loss),logs=logs,
                gradient_norm=grads,max_parameter_change=changes,seconds=time.perf_counter()-clock,
                has_category_evidence=any('img_evidence' in n or 'text_evidence' in n for n,_ in m.named_parameters()),
                claim='Implementation smoke only; no retrieval/UQ improvement claim')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()

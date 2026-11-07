"""A full-architecture CPU synthetic Adam step. Not a real INRIA evaluation."""
import argparse,json,time
from pathlib import Path
import torch
from core import EvidenceOnE,AuxConfig,objective,base
p=argparse.ArgumentParser()
p.add_argument('--task',choices=['matching','category'],default='matching')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():
    raise FileExistsError(a.output)
torch.set_num_threads(1)
torch.manual_seed(2)
m=EvidenceOnE(4096,1000,100,task=a.task)
initial=base.backbone_hash(m)
x=torch.randn(2,4096)
t=torch.randn(2,1000)
y=torch.eye(100)[:2]
watch={n:v.detach().clone() for n,v in m.named_parameters()
       if n in ('img_net.layers.4.weight','text_net.layers.4.weight')
       or 'evidence.bias' in n or 'pair_head.net.2' in n}
opt=torch.optim.Adam(m.parameters(),lr=5e-5,betas=(.5,.999))
start=time.perf_counter()
loss,logs,probe=objective(m,(x,t,y),AuxConfig(pair_chunk=2),(1.,1.),
                          epoch=10,backward=True,probe=True)
assert all(v.grad is None or torch.isfinite(v.grad).all() for v in m.parameters())
opt.step()
changes={n:float((v.detach()-watch[n]).abs().max()) for n,v in m.named_parameters()
         if n in watch}
assert all(v>0 for v in changes.values())
report=dict(task=a.task,synthetic_only=True,torch=str(torch.__version__),
            device='cpu',parameters=sum(v.numel() for v in m.parameters()),
            initial_backbone_sha256=initial,seconds=time.perf_counter()-start,
            losses=logs,output_layer_probe=probe,max_parameter_changes=changes,
            real_INRIA_or_performance_claim=False)
a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps(report,indent=2))

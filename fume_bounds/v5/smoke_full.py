"""Optional CPU full-width synthetic smoke test; NOT an INRIA experiment.

Uses the original INRIA tower dimensions, all four training terms, one real
Adam step, and reports finite gradients/updates. Requires several GB of RAM.
"""
import argparse
import json
from pathlib import Path
import resource
import time
import torch
if __package__:
    from .model import MultiLabelV5, ModelConfig
    from .losses import LossConfig, objective
else:
    from model import MultiLabelV5, ModelConfig
    from losses import LossConfig, objective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    torch.manual_seed(2)
    model = MultiLabelV5(ModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(.5,.999))
    image = torch.randn(2,4096)
    text = torch.randn(2,1000)
    labels = torch.zeros(2,100)
    labels[0,[1,3,5]] = 1
    labels[1,[2,3,8]] = 1
    watched = ('img_net.layers.4.weight', 'text_net.layers.4.weight',
               'img_evidence.bias', 'text_evidence.bias')
    params = dict(model.named_parameters())
    old = {n:params[n].detach().clone() for n in watched}
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    total, log = objective(model,(image,text,labels),LossConfig(),(1.,1.),backward=True)
    finite = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    if not finite:
        raise FloatingPointError('Nonfinite gradients')
    grad_norms = {n:float(params[n].grad.norm()) for n in watched}
    optimizer.step()
    changes = {n:float((params[n].detach()-old[n]).abs().max()) for n in watched}
    if not all(v>0 for v in changes.values()):
        raise RuntimeError('Some watched parameters did not update')
    result = dict(scope='one CPU Adam step, synthetic paired features, NOT real INRIA',
        dimensions=dict(image=4096,text=1000,hidden=4096,classes=100,layers=4,batch=2),
        parameters=sum(p.numel() for p in model.parameters()),
        loss=log, gradients_finite=finite, watched_gradient_norms=grad_norms,
        max_parameter_changes=changes, observed_step_seconds=time.perf_counter()-start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        torch_version=str(torch.__version__),cuda_available=torch.cuda.is_available())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

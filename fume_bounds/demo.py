"""Small deterministic two-encoder regression of the mechanism; NOT INRIA/FUME."""
import json
from pathlib import Path
import numpy as np
import torch
from live_reference import LiveReference, Event, RiskSystem, phase_one
from fume_control import (TransactionalController, ControlConfig, RiskGuard,
                          run_encoder_segment)

ROOT=Path(__file__).parent
OUT=ROOT/'validation';OUT.mkdir(exist_ok=True)
torch.set_num_threads(1);torch.manual_seed(7)

class TinyTwoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        self.text=torch.nn.Linear(1,1,bias=False,dtype=torch.float64)
        with torch.no_grad():self.image.weight.fill_(.3);self.text.weight.fill_(.5)
        self.register_buffer('head_offsets',torch.tensor([0.,.7],dtype=torch.float64))
    def forward(self):
        x=torch.ones(8,1,dtype=torch.float64)
        def heads(z):return torch.sigmoid(z[:,None,:]+self.head_offsets[None,:,None])
        return heads(self.image(x)),heads(self.text(x))

model=TinyTwoEncoder()
y=np.ones((8,1))
problem=LiveReference(y,anchor_alpha=[.5,.5],kappa=.005,
    events=[Event('image',0,0),Event('text',1,0)],sample_ids=[str(x) for x in range(8)],
    heads_fingerprint='fixed_two_offsets_0_and_0.7',task='multilabel',gap_tol=1e-6)
optimizer=torch.optim.Adam(model.parameters(),lr=.04)
log=OUT/'demo_attempts.jsonl'
if log.exists():log.unlink()
control=TransactionalController(model,optimizer,ControlConfig(max_attempts=8,shrink=.5),log_path=log)
guard=RiskGuard(component_budget=(0.,0.),mean_budget=0.,risk_atol=1e-10)

def evaluate():
    i,t=model()
    return problem.solve(i.detach().numpy(),t.detach().numpy()).reference_solve()

def loss(_batch):
    i,t=model()
    snap=problem.solve(i.detach().numpy(),t.detach().numpy())
    return problem.risk_barriers(i,t).mean()+problem.loss(i,t,snap)

start=evaluate().evaluation()
rounds=[]
for r in range(3):
    report=run_encoder_segment(control,replay_batches=[0,1,2,3],current_loss=loss,
        solve_current_reference=evaluate,risk_guard=guard,round_id=f'toy_{r}')
    rounds.append(report.as_dict())
    if not report.accepted:break
end=evaluate().evaluation()
# Old-family counterexample and stale-start distinction.
iold=np.tile(np.array([.2,.4])[None,:,None],(8,1,1));inew=np.tile(np.array([.8,.9])[None,:,None],(8,1,1))
t=np.full_like(iold,.6)
oldrisk=RiskSystem.from_predictions(iold,t,y);newrisk=RiskSystem.from_predictions(inew,t,y)
old_empty=phase_one(oldrisk,newrisk.risks(np.array([.5,.5]))+.01)
# Directional derivative verification from the tests, with reported numbers.
theta=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
def banks(v):
    i=torch.sigmoid(v+torch.tensor([0.,.7],dtype=torch.float64))[None,:,None].expand(8,-1,1)
    return i,torch.full_like(i,.6)+v*0.
p=LiveReference(y,anchor_alpha=[.5,.5],kappa=.005,events=[Event('image',0,0)],
                sample_ids=[str(x) for x in range(8)],heads_fingerprint='fixed',task='multilabel',gap_tol=1e-6)
i,tt=banks(theta);s=p.solve(i.detach().numpy(),tt.detach().numpy())
g=-torch.autograd.grad(p.loss(i,tt,s),theta)[0].item()
def val(v):
    a,b=banks(torch.tensor(v,dtype=torch.float64));return p.solve(a.numpy(),b.numpy()).objective_upper
eps=1e-4;fd=(val(.7+eps)-val(.7-eps))/(2*eps)
report=dict(scope='synthetic two scalar encoders; not INRIA, not FUME, no population guarantee',
            current_banks_always_used=True,
            initial_bound=[start.bound.lower,start.bound.upper],
            final_bound=[end.bound.lower,end.bound.upper],
            initial_risks=start.risks,final_risks=end.risks,
            accepted_segments=sum(r['status']=='accepted' for r in rounds),
            committed_steps=control.total_committed_optimizer_steps,
            final_parameters={n:p.detach().tolist() for n,p in model.named_parameters()},
            old_family_infeasibility=old_empty.as_dict(),
            gradient_check=dict(envelope_gradient=g,finite_difference=fd,absolute_error=abs(g-fd)),
            rounds=rounds)
(OUT/'demo_result.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='rounds'},indent=2))

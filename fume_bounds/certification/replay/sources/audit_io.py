from pathlib import Path
import json
import torch

def log_metric(epoch,phase,loss,i2t,t2i):
    with Path('curve.jsonl').open('a') as output:
        output.write(json.dumps(dict(epoch=epoch,phase=phase,loss=loss,i2t=i2t,t2i=t2i,mean=(i2t+t2i)/2),allow_nan=False)+'\n')

def save_checkpoint(epoch,model,score):
    tmp=Path('best.tmp')
    torch.save(dict(epoch=epoch,state=model.state_dict(),W=model.W.detach(),val_mean=score),tmp)
    tmp.replace('best.pt')

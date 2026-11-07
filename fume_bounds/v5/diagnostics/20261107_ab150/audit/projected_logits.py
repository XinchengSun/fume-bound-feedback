import json, sys, gc
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from independent_inria_geometry import run, runlib, build_model, ModelConfig
torch.set_num_threads(2)
data,_=runlib.checked_data()
out={}
for arm in ("A_fume","B_bce","C_category"):
    ck=torch.load(run/arm/"last.pt",mmap=True,weights_only=True,map_location="cpu")
    weights=ck.get("model_state",ck.get("model"))
    model=build_model(ModelConfig(),"category")
    keys=model.state_dict()
    keys.update(weights); model.load_state_dict(keys)
    agg={}
    with torch.inference_mode():
        for x,t,y in DataLoader(data["val"],batch_size=128):
            for name, row in zip(("img","txt"),model(x,t)):
                logits=row["logits"].double()
                frac=logits.clamp(max=0).square().sum(1)/logits.square().sum(1)
                hit_frac=(y*logits.square()).sum(1)/logits.square().sum(1)
                agg.setdefault(name,[]).append(torch.stack([frac,hit_frac],dim=1))
    out[arm]={k:dict(zip(("energy_negative_coordinates","energy_true_label_coordinate"),torch.cat(v).mean(0).tolist())) for k,v in agg.items()}
    out[arm]["W_orthogonality_max_error"]=float((model.W.T@model.W-torch.eye(100)).abs().max())
    del model,ck,weights,keys
    gc.collect()
Path(sys.argv[1]).write_text(json.dumps(out,indent=2))
print(json.dumps(out))

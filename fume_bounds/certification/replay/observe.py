"""Observe actual Adam starts inside the unchanged mechanism generator."""
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import mechanism as m


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def rng_state():
    return {k: hashlib.sha256(pickle.dumps(v)).hexdigest() for k,v in {
        "python": random.getstate(), "numpy_global": np.random.get_state(),
        "numpy_generator": m.e.rng.bit_generator.state,
        "torch_cpu": torch.random.get_rng_state().numpy().tobytes(),
    }.items()}


def main():
    model=m.e.DATA
    starts=[tensor_hash(model[k]) for k in ("wI","wT")]
    W=tensor_hash(model["W"])
    arms=[("base","original")]+[(g,k) for k in ("plain","map_logit","predictive_mean","stop_sd","full")
                                for g in ("alone","plus_FUME")]
    rows=[]
    def inspect(opt,args,kwargs):
        params=opt.param_groups[0]["params"]
        hashes=[tensor_hash(v) for v in params]
        if hashes!=starts or opt.state or tensor_hash(model["W"])!=W:
            raise RuntimeError("weights, fixed W or optimizer differ at branch start")
        state=rng_state()
        settings={k:v for k,v in opt.param_groups[0].items() if k!="params"}
        if rows and (rows[0]["rng"]!=state or rows[0]["adam"]!=settings):
            raise RuntimeError("RNG or Adam settings differ between branch starts")
        batch,arm=divmod(len(rows),len(arms))
        rows.append(dict(batch=batch,group=arms[arm][0],kind=arms[arm][1],rng=state,
                         weight_sha256=hashes,fixed_W_sha256=W,adam=settings,state_entries=len(opt.state)))
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    hook=register_optimizer_step_pre_hook(inspect)
    try:
        m.main()
    finally:
        hook.remove()
    assert len(rows)==88
    (Path(__file__).parent/"starts.json").write_text(json.dumps(rows,indent=2))

if __name__=="__main__":
    main()


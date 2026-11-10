from pathlib import Path
import json
from main import set_seed
from options import get_dataloader
from model import FUME
from evaluate import fx_calc_map_label,fx_calc_map_label_withUncertainty
import numpy as np
import torch

set_seed(2)
cfg,loaders=get_dataloader('nus_deep')
state=torch.load('best.pt',map_location='cuda',weights_only=False)
model=FUME('cuda',cfg['input_dim_I'],cfg['input_dim_T'],10,10,cfg['layer_num']).cuda().eval()
model.load_state_dict(state['state'])
model.W.copy_(state['W'])
rows=[json.loads(s) for s in Path('curve.jsonl').read_text().splitlines()]
assert [(r['epoch'],r['phase']) for r in rows]==[(i,p) for i in range(1,101) for p in ('val','test')]
choice=max((r for r in rows if r['phase']=='val'),key=lambda r:r['mean'])
assert state['epoch']==choice['epoch'] and abs(choice['mean']-state['val_mean'])<1e-12
result=dict(commit='5dd7a306aa505d8f60265b0353ff98de19ad9815',seed=2,epochs=100,config=cfg,selected_epoch=choice['epoch'],sizes={k:len(v.dataset) for k,v in loaders.items()})
with torch.inference_mode():
    for split in ('val','test'):
        images,texts,labels=[],[],[]
        for x,t,y in loaders[split]:
            pred=model(x.cuda(),t.cuda())
            images.append(pred['view1_feature'].cpu().numpy())
            texts.append(pred['view2_feature'].cpu().numpy())
            labels.append(y.numpy())
        i,t,y=np.concatenate(images),np.concatenate(texts),np.concatenate(labels).argmax(1)
        assert len(y)==1000
        a=fx_calc_map_label(i,t,y,0)
        b=fx_calc_map_label(t,i,y,0)
        scores=dict(i2t=a,t2i=b,mean=(a+b)/2)
        expected=next(r for r in rows if r['epoch']==choice['epoch'] and r['phase']==split)
        assert all(abs(value-expected[k])<1e-12 for k,value in scores.items())
        result[split]=scores
        if split=='test':
            saved=next(Path('saved/nus_deep').iterdir())
            for file,arr in [('I_feat.npy',i),('T_feat.npy',t)]:
                assert np.array_equal(np.load(saved/file),arr)
            filtered=[fx_calc_map_label_withUncertainty(x,z,y,np.load(saved/name),0,.5) for x,z,name in [(i,t,'I2T_uncer.npy'),(t,i,'T2I_uncer.npy')]]
            result['test_threshold_0.5']=dict(i2t=filtered[0],t2i=filtered[1],mean=np.mean(filtered))
result['replay_agrees']=True
Path('result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result),flush=True)

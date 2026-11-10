from pathlib import Path
import hashlib,json,os,sys
import torch
import numpy as np
ROOT=Path(os.environ['FUME_REPLAY_ROOT'])
HERE=Path(__file__).resolve().parent
OFFICIAL=ROOT/'2025-CVPR-FUME'
RUN=ROOT/'runs/nus_official_seed2_replay_20261108'
sys.path.insert(0,str(OFFICIAL))
from model import FUME
from options import get_dataloader
from main import set_seed
from evaluate import fx_calc_map_label

def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    torch.set_num_threads(3)
    os.chdir(RUN)
    set_seed(2)
    cfg,loaders=get_dataloader('nus_deep')
    with torch.serialization.safe_globals([np._core.multiarray.scalar,np.dtype,type(np.dtype('float64'))]):
        ck=torch.load('best.pt',map_location='cpu',weights_only=True)
    model=FUME('cpu',4096,300,10,10,4).eval()
    model.load_state_dict(ck['state'])
    model.W.copy_(ck['W'])
    expected=json.loads(Path('result.json').read_text())
    assert ck['epoch']==expected['selected_epoch']==6
    result={'checkpoint_sha256':digest('best.pt'),'epoch':ck['epoch'],'dataset_sha256':digest('datasets/nus_wide-10k_deep_doc2vec-corr-ae.h5py'),'official_model_sha256':digest(OFFICIAL/'model.py')}
    pack={'W':ck['W'],'wI':model.img_net.layers[-1].weight.detach(),'wT':model.text_net.layers[-1].weight.detach()}
    for split,loader in loaders.items():
        ds=loader.dataset
        labels=torch.as_tensor(ds.labels).argmax(1)
        assert torch.as_tensor(ds.labels).sum(1).eq(1).all()
        pen=[]
        with torch.no_grad():
            for net,data in zip((model.img_net,model.text_net),(ds.I,ds.T)):
                rows=[]
                for x in torch.as_tensor(data).float().split(128):
                    for layer in net.layers[:-1]:x=layer(x).relu()
                    rows.append(x)
                pen.append(torch.cat(rows))
            pack[split]=(*pen,labels)
            if split!='train':
                z=[torch.nn.functional.normalize(x@w.T).numpy() for x,w in zip(pen,[pack['wI'],pack['wT']])]
                v=[float(fx_calc_map_label(z[i],z[1-i],labels.numpy(),0)) for i in (0,1)]
                result[split+'_cpu']=v
                assert max(abs(v[i]-expected[split][k]) for i,k in enumerate(('i2t','t2i')))<1e-5
    torch.save(pack,HERE/'features.pt')
    (HERE/'provenance.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)

if __name__=='__main__':main()


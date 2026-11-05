import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from vls_transfer import (
    VLSConfig, FUMEVLS, PairEvidenceHead, matching_pairs, vicinal_pairs,
    noise_pairs, edl_expected_cross_entropy,
)
from train_fume_vls import map_all_queries


class TinyFUME(nn.Module):
    def __init__(self, di=7, dt=5, d=3):
        super().__init__()
        self.img_net = nn.Sequential(nn.Linear(di,8), nn.ReLU(), nn.Linear(8,d))
        self.text_net = nn.Sequential(nn.Linear(dt,8), nn.ReLU(), nn.Linear(8,d))

    def forward(self, img, txt):
        zi, zt = self.img_net(img), self.text_net(txt)
        return dict(view1_feature=zi, view2_feature=zt,
                    view1_membershipDegree=zi.softmax(-1),
                    view2_membershipDegree=zt.softmax(-1))


def dummy_fume_loss(zi, zt, mi, mt, y1, y2, weight):
    return ((zi-zt)**2).mean()*weight + ((mi-y1.float())**2).mean() + ((mt-y2.float())**2).mean()


def sample():
    torch.manual_seed(12)
    img = torch.randn(12,7)
    txt = torch.randn(12,5)
    y = F.one_hot(torch.arange(12) % 3,3)
    return img,txt,y


def test_pairs_preserve_all_aligned_positive_rows():
    _,_,y = sample()
    i,t,r = matching_pairs(y,2)
    assert torch.equal(i[:len(y)],torch.arange(len(y)))
    assert torch.equal(t[:len(y)],torch.arange(len(y)))
    assert torch.all(r[:len(y)]==1)
    assert r[len(y):].sum()==0
    for a,b in zip(i[len(y):],t[len(y):]):
        assert not bool((y[a] & y[b]).any())


def test_no_negatives_keeps_positives():
    y=torch.ones(6,1)
    i,t,r=matching_pairs(y,3)
    assert len(i)==6 and torch.all(r==1)


def test_cross_view_pair_mix_uses_same_fraction():
    torch.manual_seed(3)
    x=torch.arange(10,dtype=torch.float).unsqueeze(1)
    t=3*x+7
    y=torch.tensor([0.,1.]*5)
    xm,tm,ym=vicinal_pairs(x,t,y,beta=10)
    assert torch.allclose(tm,3*xm+7,atol=1e-6)
    assert (ym>=0).all() and (ym<=1).all()


def test_noise_target_is_soft_uniform_blend():
    torch.manual_seed(7)
    x=torch.randn(400,4)
    t=torch.randn(400,2)
    y=torch.ones(400)
    _,_,target=noise_pairs(x,t,y,1,1,1)
    assert ((target>=.5)&(target<=1)).all()
    assert .62 < target.mean() < .88


def test_original_edl_digamma_expected_ce():
    logits=torch.zeros(10,2,requires_grad=True)
    y=torch.linspace(0,1,10)
    loss=edl_expected_cross_entropy(logits,y)
    a=1+F.softplus(torch.zeros(10,2))
    expected=torch.digamma(a.sum(-1))[0]-torch.digamma(a[0,0])
    assert torch.allclose(loss,expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_invalid_config_caught():
    with pytest.raises(ValueError): VLSConfig(mix_beta=0).validate()
    with pytest.raises(ValueError): VLSConfig(vicinal_weight=0,noise_weight=0,clean_weight=0).validate()


def test_gradients_reach_both_fume_towers_and_evidence_head():
    img, txt, y=sample()
    model=FUMEVLS(TinyFUME(),3)
    total, logs=model.forward_losses(img,txt,y,dummy_fume_loss,.05)
    total.backward()
    assert torch.isfinite(total)
    for module in (model.fume.img_net,model.fume.text_net,model.head):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    assert logs['pairs'] >= 12 and logs['negatives'] > 0


def test_each_loss_toggle_is_functional():
    img,txt,y=sample()
    for cfg in (VLSConfig(vicinal_weight=1,noise_weight=0),
                VLSConfig(vicinal_weight=0,noise_weight=1),
                VLSConfig(vicinal_weight=0,noise_weight=0,clean_weight=1)):
        model=FUMEVLS(TinyFUME(),3,config=cfg)
        loss,logs=model.forward_losses(img,txt,y,dummy_fume_loss,0.05)
        assert torch.isfinite(loss)
        assert 'vls' in logs


def test_train_actual_adam_step_changes_both_encoders():
    img,txt,y=sample()
    model=FUMEVLS(TinyFUME(),3)
    optim=torch.optim.Adam(model.parameters(),lr=0.002)
    a=model.fume.img_net[0].weight.detach().clone()
    b=model.fume.text_net[0].weight.detach().clone()
    loss,_=model.forward_losses(img,txt,y,dummy_fume_loss,.05)
    loss.backward();optim.step()
    assert not torch.equal(a,model.fume.img_net[0].weight)
    assert not torch.equal(b,model.fume.text_net[0].weight)


def test_cosine_map_full_queries_and_zero_relevant():
    img=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    txt=img.clone()
    y=torch.tensor([[1,0],[0,1],[0,0]])
    m=map_all_queries(img,txt,y,chunk=2)
    assert abs(m['i2t_map_all']-2/3)<1e-6
    assert abs(m['t2i_map_all']-2/3)<1e-6


def test_pair_head_requires_matched_dimension():
    h=PairEvidenceHead(4)
    with pytest.raises(ValueError): h(torch.zeros(3,4),torch.zeros(2,4))


def test_multilabel_pair_relevance():
    y=torch.tensor([[1,1,0],[0,1,0],[0,0,1]])
    i,t,truth=matching_pairs(y,1)
    for a,b,lab in zip(i,t,truth):
        if lab.item()==0:
            assert not bool((y[a]*y[b]).any())


def test_score_is_finite_for_strong_mixture():
    torch.manual_seed(55)
    img,txt,y=sample()
    model=FUMEVLS(TinyFUME(),3,config=VLSConfig(mix_beta=20,noise_std=2))
    loss,_=model.forward_losses(img,txt,y,dummy_fume_loss,.05)
    assert torch.isfinite(loss)


def test_command_line_full_training_on_mock_fume_repository(tmp_path):
    import subprocess
    import sys as python_sys
    import json
    upstream = tmp_path / 'upstream'
    upstream.mkdir()
    (upstream/'model.py').write_text('''import torch\nfrom torch import nn\nclass FUME(nn.Module):\n def __init__(self,device,img_input_dim,text_input_dim,output_dim,num_class,layer_num):\n  super().__init__()\n  self.img_net=nn.Linear(img_input_dim,output_dim)\n  self.text_net=nn.Linear(text_input_dim,output_dim)\n  self.W=torch.eye(output_dim,num_class).to(device)\n def forward(self,img,text):\n  a=self.img_net(img); b=self.text_net(text)\n  return {"view1_feature":a,"view2_feature":b,"view1_membershipDegree":a.softmax(-1),"view2_membershipDegree":b.softmax(-1)}\n''')
    (upstream/'processor.py').write_text('''import torch\ndef calc_loss(a,b,pa,pb,ya,yb,alpha):\n return ((pa-ya.float())**2).mean()+((pb-yb.float())**2).mean()+alpha*((a-b)**2).mean()\n''')
    (upstream/'options.py').write_text('''import torch\nfrom torch.utils.data import DataLoader,TensorDataset\ndef get_dataloader(name):\n torch.manual_seed(42)\n def make(n):\n  labels=torch.nn.functional.one_hot(torch.arange(n)%3,3)\n  x=torch.randn(n,4);t=torch.randn(n,4)\n  return DataLoader(TensorDataset(x,t,labels),batch_size=6,shuffle=False)\n return dict(input_dim_I=4,input_dim_T=4,class_number=3,layer_num=1,alpha=.05),dict(train=make(24),test=make(12))\n''')
    output = tmp_path / 'result'
    script = pathlib.Path(__file__).resolve().parents[1] / 'train_fume_vls.py'
    process=subprocess.run([python_sys.executable,str(script), '--fume-dir',str(upstream),
                            '--dataset','INRIA','--epochs','1','--batch-size','6',
                            '--device','cpu','--threads','1','--output',str(output)],
                           text=True,capture_output=True,timeout=40)
    assert process.returncode == 0, process.stderr + process.stdout
    assert (output/'vls_final.pt').is_file()
    report=json.loads((output/'test_metrics.json').read_text())
    assert 0 <= report['i2t_map_all'] <= 1
    assert 0 <= report['t2i_map_all'] <= 1
    history=[json.loads(line) for line in (output/'history.jsonl').read_text().splitlines()]
    assert len(history)==1 and history[0]['training_samples']==24
    assert 'vicinal' in history[0] and 'noise' in history[0]
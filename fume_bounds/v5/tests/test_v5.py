import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from v5.model import ModelConfig, MultiLabelV5, warmstart_fume
from v5.losses import (LossConfig, classification_loss, expected_ce, fume_contrastive,
                       mix_batch, noise_batch, objective)
from v5.data import load_data
from v5.metrics import (beta_statistics, detection, risk_coverage, retrieval_map, evaluate)

torch.set_num_threads(1)


def model():
    torch.manual_seed(5)
    return MultiLabelV5(ModelConfig(5, 6, 4, hidden=12, layers=2))


def batch(n=7):
    torch.manual_seed(6)
    x, t = torch.randn(n, 5), torch.randn(n, 6)
    y = F.one_hot(torch.arange(n) % 4, 4).float()
    y[0] = torch.tensor([1., 1., 1., 0.])
    return x, t, y


def old_contrastive(zi, zt):
    # Verbatim arithmetic of upstream processor.py consistency_Learning, tau=1.
    fea, batch_size, n_view = [zi, zt], len(zi), 2
    all_fea = torch.cat(fea)
    sim = all_fea.mm(all_fea.t()).exp()
    sim = sim-sim.diag().diag()
    sim_sum1 = sum([sim[:, v*batch_size:(v+1)*batch_size] for v in range(n_view)])
    diag1 = torch.cat([sim_sum1[v*batch_size:(v+1)*batch_size].diag() for v in range(n_view)])
    loss1 = -(diag1/sim.sum(1)).log().mean()
    sim_sum2 = sum([sim[v*batch_size:(v+1)*batch_size] for v in range(n_view)])
    diag2 = torch.cat([sim_sum2[:, v*batch_size:(v+1)*batch_size].diag() for v in range(n_view)])
    loss2 = -(diag2/sim.sum(1)).log().mean()
    return loss1+loss2


def test_shapes_unit_prior_and_no_fuzzy_state():
    m = model(); x, t, _ = batch()
    for out in m(x, t):
        assert out['z'].shape == (7, 4)
        assert torch.allclose(out['z'].norm(dim=-1), torch.ones(7), atol=1e-6)
        assert out['evidence_total'].shape == (7, 1)
        assert torch.all(out['alpha'] >= 1) and torch.all(out['beta'] >= 1)
        assert torch.allclose(out['alpha']+out['beta'], out['evidence_total']+2)
        assert torch.allclose(out['predictive_logits'].sigmoid(), out['probability'], atol=1e-7)
    assert 'W' in m.state_dict() and 'W' not in dict(m.named_parameters())
    assert not any('fuzzy' in k or 'membership' in k or 'head.net' in k for k in m.state_dict())
    assert sum(p.numel() for n,p in m.named_parameters() if 'evidence' in n) == 2*(12+1)


def test_zero_evidence_is_beta_1_1():
    m = model()
    with torch.no_grad():
        m.img_evidence.bias.fill_(-1000)
    out = m.view(batch()[0], 'image')
    assert torch.equal(out['alpha'], torch.ones(7, 4))
    assert torch.equal(out['beta'], torch.ones(7, 4))
    assert torch.equal(out['probability'], torch.full((7, 4), .5))
    s = beta_statistics(out['alpha'], out['beta'])
    assert torch.allclose(s['total_entropy'], torch.full_like(s['total_entropy'], np.log(2)))
    assert torch.allclose(s['aleatoric_entropy'], torch.full_like(s['aleatoric_entropy'], .5))
    assert torch.equal(s['vacuity'], torch.ones_like(s['vacuity']))


def test_multihot_can_predict_several_ones_without_class_normalization():
    a = torch.tensor([[12., 12., 12., -12.]])
    s = torch.tensor([[1000.]])
    alpha, beta = 1+s*a.sigmoid(), 1+s*(1-a.sigmoid())
    p = alpha/(alpha+beta)
    assert (p[0,:3] > .99).all() and p[0,3] < .01
    assert p.sum() > 2.9
    target = torch.tensor([[1.,1.,1.,0.]])
    logits = alpha.log()-beta.log()
    assert classification_loss({'predictive_logits':logits},target) < .01


def test_classification_uses_final_prior_shrunk_mean():
    a = torch.tensor([[2., -2., 0., 1.]], dtype=torch.float64)
    s = torch.tensor([[.3]], dtype=torch.float64)
    alpha, beta = 1+s*a.sigmoid(), 1+s*(1-a.sigmoid())
    p = alpha/(alpha+beta)
    y = torch.tensor([[1.,0.,1.,0.]], dtype=torch.float64)
    actual = classification_loss({'predictive_logits':alpha.log()-beta.log()}, y)
    direct = -(y*p.log()+(1-y)*torch.log1p(-p)).mean()
    assert actual.item() == pytest.approx(direct.item(), abs=1e-7)
    assert not torch.allclose(p, a.sigmoid())


@pytest.mark.parametrize('n', [1,2,7,12])
def test_contrastive_values_and_gradients_match_fume(n):
    torch.manual_seed(n)
    xi = torch.randn(n, 4, dtype=torch.float64, requires_grad=True)
    xt = torch.randn(n, 4, dtype=torch.float64, requires_grad=True)
    zi,zt = F.normalize(xi,dim=-1), F.normalize(xt,dim=-1)
    old, new = old_contrastive(zi,zt), fume_contrastive(zi,zt)
    assert torch.allclose(old,new,atol=1e-12)
    a = torch.autograd.grad(old,(xi,xt),retain_graph=True)
    b = torch.autograd.grad(new,(xi,xt))
    assert all(torch.allclose(x,y,atol=1e-12) for x,y in zip(a,b))


def test_expected_ce_uses_labelwise_beta_not_categorical_sum():
    a=torch.tensor([[2.,4.,6.],[9.,2.,3.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[5.,3.,2.],[2.,7.,6.]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[1.,1.,0.],[.2,.7,.8]],dtype=torch.float64)
    actual=expected_ce({'alpha':a,'beta':b},t)
    manual=sum((t[i,k]*(torch.digamma(a[i,k]+b[i,k])-torch.digamma(a[i,k]))+
                (1-t[i,k])*(torch.digamma(a[i,k]+b[i,k])-torch.digamma(b[i,k])))
                for i in range(2) for k in range(3))/6
    assert torch.allclose(actual,manual,atol=1e-12)
    grads=torch.autograd.grad(actual,(a,b))
    assert torch.allclose(grads[0],(torch.polygamma(1,a+b)-t*torch.polygamma(1,a))/6)
    assert torch.allclose(grads[1],(torch.polygamma(1,a+b)-(1-t)*torch.polygamma(1,b))/6)


def test_mixup_has_same_indices_lambda_and_multihot_targets():
    y=torch.tensor([[1.,1.,1.,0.],[0.,1.,0.,1.],[1.,0.,0.,1.]])
    xi,xt,target=mix_batch(y,y.clone(),y)
    assert torch.equal(xi,xt) and torch.equal(xi,target)
    assert (target.sum(1)>1).all()  # not converted to class probabilities
    assert ((target>=0)&(target<=1)).all()


def test_noise_target_is_half_per_label_not_one_over_classes():
    y=torch.tensor([[1.,1.,1.,0.],[0.,1.,0.,1.]])
    torch.manual_seed(2)
    xi,xt,target=noise_batch(y,y.clone(),y,(0.,0.))
    assert torch.equal(xi,xt)
    for i in range(2):
        lam=xi[i,y[i]==1][0]
        assert torch.allclose(target[i],lam*y[i]+(1-lam)*.5)


def test_staged_backward_equals_joint_backward_and_updates_both_towers():
    a=model(); b=copy.deepcopy(a); inputs=batch()
    cfg=LossConfig(); scales=(.8,1.2)
    torch.manual_seed(88)
    loss_a,_=objective(a,inputs,cfg,scales,backward=False);loss_a.backward()
    torch.manual_seed(88)
    loss_b,_=objective(b,inputs,cfg,scales,backward=True)
    assert loss_a.item()==pytest.approx(loss_b.item(),rel=1e-6)
    for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert p.grad is not None,n
        assert torch.allclose(p.grad,q.grad,atol=2e-7,rtol=1e-5),n
    before={n:p.detach().clone() for n,p in b.named_parameters()}
    w=b.W.clone()
    torch.optim.Adam(b.parameters(),lr=1e-3,betas=(.5,.999)).step()
    for prefix in ('img_net.','text_net.','img_evidence.','text_evidence.'):
        assert any(not torch.equal(p,before[n]) for n,p in b.named_parameters() if n.startswith(prefix))
    assert torch.equal(w,b.W)


@pytest.mark.parametrize('k',[2,4,10])
def test_beta_statistics_matches_distribution_and_sampling(k):
    torch.manual_seed(k)
    a=1+torch.rand(2,k,dtype=torch.float64)*10
    b=1+torch.rand(2,k,dtype=torch.float64)*10
    out=beta_statistics(a,b)
    distribution=torch.distributions.Beta(a,b)
    assert torch.allclose(out['beta_entropy'],distribution.entropy(),atol=1e-12)
    assert torch.allclose(out['total_entropy'],out['aleatoric_entropy']+out['epistemic_mi'])
    draws=distribution.sample((30000,))
    h=-draws*draws.log()-(1-draws)*torch.log1p(-draws)
    assert torch.allclose(h.mean(0),out['aleatoric_entropy'],atol=.004)


def test_same_mean_higher_evidence_lower_epistemic():
    out=beta_statistics(torch.tensor([[1.,50.]]),torch.tensor([[1.,50.]]))
    assert out['probability'][0,0]==out['probability'][0,1]
    assert out['epistemic_mi'][0,1]<out['epistemic_mi'][0,0]
    assert out['aleatoric_entropy'][0,1]>out['aleatoric_entropy'][0,0]


@pytest.mark.parametrize('setting', [ModelConfig(hidden=0),ModelConfig(initial_evidence=float('nan')),
                                   ModelConfig(norm_eps=0), ModelConfig(classes=1)])
def test_invalid_model_config_rejected(setting):
    with pytest.raises(ValueError): MultiLabelV5(setting)


@pytest.mark.parametrize('setting', [LossConfig(vicinal=float('nan')), LossConfig(noise=-1),
                                    LossConfig(temperature=0),LossConfig(0,0,0,0)])
def test_invalid_loss_config_rejected(setting):
    with pytest.raises(ValueError): setting.validate()


def write_data(path, index=False):
    rng=np.random.default_rng(80);arrays={}
    for split,n in [('train',13),('val',6),('test',7)]:
        y=np.eye(4,dtype=np.float32)[np.arange(n)%4]
        if not index:y[0]=[1,1,1,0]
        arrays[split+'_image']=rng.normal(size=(n,5)).astype(np.float32)
        arrays[split+'_text']=rng.normal(size=(n,6)).astype(np.float32)
        arrays[split+'_labels']=(np.argmax(y,axis=1)+1).reshape(-1,1) if index else y
    np.savez(path,**arrays)
    return arrays


def test_npz_preserves_multihot_splits_and_train_only_noise_scale(tmp_path):
    arrays=write_data(tmp_path/'data.npz')
    data,receipt=load_data(tmp_path/'data.npz')
    assert receipt['classes']==4
    assert torch.equal(data['train'].tensors[2][0],torch.tensor([1.,1.,1.,0.]))
    for k in ('train','val','test'):
        assert len(data[k])==len(arrays[k+'_image'])
    expected=float(np.sqrt(np.mean(arrays['train_image'].astype(float)**2)))
    assert receipt['train_noise_rms'][0]==pytest.approx(expected)


def test_original_mat_keys_and_explicit_class_index_base(tmp_path):
    arrays=write_data(tmp_path/'index.npz',index=True)
    mapping={'train':('tr_img','tr_txt','tr_img_lab'), 'val':('val_img','val_txt','val_img_lab'),
             'test':('te_img','te_txt','te_img_lab')}
    original={}
    for s,keys in mapping.items():
        for name,key in zip(('image','text','labels'),keys):original[key]=arrays[s+'_'+name]
    savemat(tmp_path/'INRIA.mat',original)
    with pytest.raises(ValueError,match='label-base'):load_data(tmp_path/'INRIA.mat')
    data,receipt=load_data(tmp_path/'INRIA.mat',classes=4,label_base=1)
    assert data['train'].tensors[2].shape==(13,4)
    assert torch.equal(data['train'].tensors[2].sum(1),torch.ones(13))


def test_bad_labels_are_rejected_not_argmaxed_or_normalized(tmp_path):
    a=write_data(tmp_path/'data.npz')
    a['train_labels'][1]=[.5,.5,0,0]
    np.savez(tmp_path/'bad.npz',**a)
    with pytest.raises(ValueError):load_data(tmp_path/'bad.npz')
    a['train_labels'][1]=0
    np.savez(tmp_path/'empty.npz',**a)
    with pytest.raises(ValueError,match='unlabeled'):load_data(tmp_path/'empty.npz')


def test_retrieval_multilabel_overlap_and_ties_against_independent_ap():
    z=torch.tensor([[1.,0.],[0.,1.],[1.,1.],[1.,0.]])
    y=torch.tensor([[1,0,0],[0,1,0],[1,1,0],[1,0,1]])
    result=retrieval_map(z,z,y,chunk=2)
    score=F.normalize(z.float(),dim=-1)@F.normalize(z.float(),dim=-1).T
    aps=[]
    for i in range(4):
        order=sorted(range(4),key=lambda j:(-float(score[i,j]),j))
        relevant=[j for j in order if (y[i]*y[j]).sum()>0]
        aps.append(sum((k+1)/(order.index(j)+1) for k,j in enumerate(relevant))/len(relevant))
    assert result['i2t_map_all']==pytest.approx(np.mean(aps),abs=1e-7)
    assert result['queries']==4 and not result['confidence_filter']


def test_detection_and_coverage_handle_ties():
    assert detection([0,1,0,1],[.5]*4)['auroc']==.5
    assert detection([1,1],[.1,.2])['auroc'] is None
    assert risk_coverage([0,0,1,1],[.5]*4)['aurc']==.5
    assert risk_coverage([1,1,0,0],[.5]*4)['aurc']==.5
    assert detection([0,0,1,1],[0,0,1,1])['ap']==1


def test_evaluation_keeps_all_rows_and_does_not_mutate_weights(tmp_path):
    m=model();x,t,y=batch();state={k:v.clone() for k,v in m.state_dict().items()}
    dataset=torch.utils.data.TensorDataset(x,t,y)
    report=evaluate(m,dataset,tmp_path/'eval',batch_size=3,chunk=2)
    assert m.training
    assert report['classification']['image']['n_samples']==7
    assert report['classification']['image']['n_labels']==4
    assert report['retrieval']['queries']==7
    for k,v in m.state_dict().items():assert torch.equal(v,state[k])
    with np.load(tmp_path/'eval'/'image_predictions.npz') as f:
        assert np.array_equal(f['labels'],y.numpy())
        assert np.all(f['alpha']>=1) and np.all(f['beta']>=1)
    with pytest.raises(FileExistsError):evaluate(m,dataset,tmp_path/'eval')


def test_fume_warmstart_strictly_restores_weights_and_W():
    m=model();state={k:v.clone() for k,v in m.state_dict().items() if k.startswith(('img_net.','text_net.'))}
    cp={'state_dict':state,'fixed_W':m.W.clone()}
    torch.manual_seed(100);other=MultiLabelV5(m.config)
    warmstart_fume(other,cp)
    for k,v in state.items():assert torch.equal(v,other.state_dict()[k])
    assert torch.equal(other.W,m.W)
    with pytest.raises(ValueError):warmstart_fume(other,{'state_dict':state})


def run(script,*args):
    completed=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],capture_output=True,text=True,timeout=90)
    assert completed.returncode==0,completed.stdout+completed.stderr
    return completed


def test_cli_train_evaluate_resume_are_reproducible(tmp_path):
    data=tmp_path/'data.npz';write_data(data)
    settings=['--data',data,'--hidden',12,'--layers',2,'--batch-size',5,'--device','cpu','--threads',1]
    run('train.py',*settings,'--epochs',1,'--output',tmp_path/'first')
    first=torch.load(tmp_path/'first'/'last.pt',weights_only=True)
    assert first['model_state']['W'].shape==(4,4)
    assert first['epoch']==1
    assert json.loads((tmp_path/'first'/'history.jsonl').read_text())['train_rows']==13
    run('evaluate.py','--data',data,'--checkpoint',tmp_path/'first'/'last.pt','--split','test',
        '--output',tmp_path/'replay','--device','cpu','--threads',1,'--batch-size',3,'--chunk',2)
    for name in ('image','text'):
        with np.load(tmp_path/'first'/'test'/f'{name}_predictions.npz') as a,np.load(tmp_path/'replay'/f'{name}_predictions.npz') as b:
            for key in a.files:assert np.allclose(a[key],b[key],atol=1e-6),key
    run('train.py','--data',data,'--resume',tmp_path/'first'/'last.pt','--epochs',2,
        '--output',tmp_path/'resumed','--device','cpu','--threads',1,'--no-test')
    run('train.py',*settings,'--epochs',2,'--output',tmp_path/'continuous','--no-test')
    resumed=torch.load(tmp_path/'resumed'/'last.pt',weights_only=True)
    continuous=torch.load(tmp_path/'continuous'/'last.pt',weights_only=True)
    for key,value in resumed['model_state'].items():
        assert torch.equal(value,continuous['model_state'][key]),key
    assert resumed['optimizer']['state'].keys()==continuous['optimizer']['state'].keys()


def test_cli_refuses_old_checkpoint_and_existing_outputs(tmp_path):
    data=tmp_path/'data.npz';write_data(data)
    cp=tmp_path/'old.pt';torch.save({'state_dict':{},'fixed_W':torch.eye(4)},cp)
    p=subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--data',str(data),'--checkpoint',str(cp),
                     '--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert p.returncode!=0 and 'v5 checkpoint' in p.stderr
    output=tmp_path/'existing';output.mkdir();(output/'keep').write_text('keep')
    p=subprocess.run([sys.executable,str(ROOT/'train.py'),'--data',str(data),'--output',str(output)],
                     capture_output=True,text=True)
    assert p.returncode!=0 and 'never overwrite' in p.stderr
    assert (output/'keep').read_text()=='keep'

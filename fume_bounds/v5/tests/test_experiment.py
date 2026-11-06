"""Regression tests for the optional experiment runner, not performance claims."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model import ModelConfig, MultiLabelV5
from losses import LossConfig, objective
from experiment import ExperimentModel, ExperimentLoss, experiment_objective, wrong_evidence_kl, belief_consistency
from run_experiment import (isolated_evaluate, learning_rate, resolved_config, parser,
                            rng_state, restore_rng, validate_config)
from data import load_data
from metrics import beta_statistics

torch.set_num_threads(1)


def make_model(mode='shared'):
    torch.manual_seed(5)
    return ExperimentModel(ModelConfig(5,6,4,hidden=12,layers=2),mode)


def sample(n=7):
    gen=torch.Generator().manual_seed(13)
    y=F.one_hot(torch.arange(n)%4,4).float();y[0]=torch.tensor([1.,1.,1.,0.])
    return torch.randn(n,5,generator=gen),torch.randn(n,6,generator=gen),y


def write_data(path):
    rng=np.random.default_rng(19);arrays={}
    for name,n in [('train',13),('val',6),('test',7)]:
        arrays[name+'_image']=rng.normal(size=(n,5)).astype('float32')
        arrays[name+'_text']=rng.normal(size=(n,6)).astype('float32')
        y=np.eye(4,dtype='float32')[np.arange(n)%4];y[0]=[1,1,1,0]
        arrays[name+'_labels']=y
    np.savez(path,**arrays)
    return arrays


def run(*args):
    p=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),*map(str,args)],capture_output=True,text=True,timeout=90)
    assert p.returncode==0,p.stdout+p.stderr
    return p


@pytest.mark.parametrize('mode',['shared','per_label'])
def test_prior_shapes_and_multihot(mode):
    m=make_model(mode);o=m(*sample()[:2])[0]
    assert o['evidence_total'].shape==(7,1 if mode=='shared' else 4)
    assert (o['alpha']>=1).all() and (o['beta']>=1).all()
    assert torch.allclose(o['alpha']+o['beta'],o['evidence_total']+2)
    assert torch.allclose(o['predictive_logits'].sigmoid(),o['probability'],atol=1e-7)
    assert torch.allclose(o['z'].norm(dim=-1),torch.ones(7),atol=1e-6)
    with torch.no_grad():m.img_evidence.bias.fill_(-1000)
    z=m.view(sample()[0],'image')
    assert torch.equal(z['probability'],torch.full((7,4),.5))
    assert torch.equal(z['alpha'],torch.ones(7,4))


def test_architecture_ablation_preserves_towers_and_rng():
    a=make_model();ra=torch.get_rng_state()
    b=make_model('per_label');rb=torch.get_rng_state()
    assert torch.equal(ra,rb)
    for k,v in a.state_dict().items():
        if 'evidence' not in k:assert torch.equal(v,b.state_dict()[k])
    assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==2*(12+1)*3
    assert 'W' not in dict(a.named_parameters())


def test_shared_checkpoint_is_exact_legacy_model():
    a=make_model();b=MultiLabelV5(a.config);b.load_state_dict(a.state_dict())
    for oa,ob in zip(a(*sample()[:2]),b(*sample()[:2])):
        for k in oa:assert torch.equal(oa[k],ob[k]),k


@pytest.mark.parametrize('epoch',[1,5,10,100])
def test_disabled_extensions_equal_legacy_loss_and_gradients(epoch):
    a=make_model();b=copy.deepcopy(a);data=sample()
    torch.manual_seed(42);la,_=objective(a,data,LossConfig(),(.8,1.2));la.backward()
    torch.manual_seed(42);lb,_=experiment_objective(b,data,ExperimentLoss(),(.8,1.2),epoch=epoch);lb.backward()
    assert torch.equal(la,lb)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert torch.equal(p.grad,q.grad),name


@pytest.mark.parametrize('mode',['shared','per_label'])
def test_all_extensions_staged_equals_joint_and_updates(mode):
    a=make_model(mode);b=copy.deepcopy(a);data=sample()
    cfg=ExperimentLoss(error_kl=.1,consistency=.02,image_noise_multiplier=.8,text_noise_multiplier=.5)
    torch.manual_seed(41);la,_=experiment_objective(a,data,cfg,(.8,1.2),epoch=10);la.backward()
    torch.manual_seed(41);lb,logs=experiment_objective(b,data,cfg,(.8,1.2),epoch=10,backward=True)
    assert la.item()==pytest.approx(lb.item(),abs=1e-7)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=1e-5),name
    before={k:v.clone() for k,v in b.state_dict().items()}
    torch.optim.Adam(b.parameters(),lr=1e-3).step()
    for prefix in ('img_net.','text_net.','img_evidence.','text_evidence.'):
        assert any(not torch.equal(v,before[k]) for k,v in b.state_dict().items() if k.startswith(prefix))
    assert torch.equal(b.W,before['W'])
    assert logs['kl_coefficient']==.1
    assert sum(v for k,v in logs.items() if k.startswith('weighted_'))==pytest.approx(logs['total'],abs=1e-6)


def test_kl_matches_torch_distribution_and_masks_correct_evidence():
    a=torch.tensor([[7.,8.,3.,1.]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[2.,3.,6.,1.]],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([[1.,0.,1.,0.]])
    v=wrong_evidence_kl({'alpha':a,'beta':b},y)
    aa=y+(1-y)*a;bb=(1-y)+y*b
    exact=torch.distributions.kl_divergence(torch.distributions.Beta(aa,bb),torch.distributions.Beta(torch.ones_like(a),torch.ones_like(b))).mean()
    assert torch.allclose(v,exact,atol=1e-12)
    ga,gb=torch.autograd.grad(v,(a,b))
    assert ga[0,0]==ga[0,2]==0 and gb[0,1]==gb[0,3]==0
    assert ga[0,1]>0 and gb[0,0]>0
    for q in [1.,1.+1e-9,1e4,1e10]:
        loss=wrong_evidence_kl({'alpha':torch.full((1,2),q,dtype=torch.float64),'beta':torch.ones(1,2)},torch.zeros(1,2))
        assert loss>=0 and torch.isfinite(loss)


def test_kl_refuses_soft_targets_and_bad_parameters():
    out={'alpha':torch.ones(1,2),'beta':torch.ones(1,2)}
    with pytest.raises(ValueError):wrong_evidence_kl(out,torch.tensor([[.6,.4]]))
    with pytest.raises(ValueError):wrong_evidence_kl({'alpha':torch.zeros(1,2),'beta':torch.ones(1,2)},torch.zeros(1,2))


@pytest.mark.parametrize('n,expected',[(0,[.1,.1,.1]),(1,[.1,.1,.1]),(3,[0.,.05,.1])])
def test_kl_annealing(n,expected):
    c=ExperimentLoss(error_kl=.1,kl_anneal_epochs=n)
    assert [c.kl_coefficient(e) for e in [1,2,3]]==pytest.approx(expected)
    assert c.kl_coefficient(100)==.1
    with pytest.raises(ValueError):c.kl_coefficient(0)


def test_belief_consistency_not_mean_only():
    a={'alpha':torch.tensor([[2.,2.]],requires_grad=True),'beta':torch.tensor([[2.,2.]],requires_grad=True)}
    b={'alpha':torch.tensor([[20.,20.]],requires_grad=True),'beta':torch.tensor([[20.,20.]],requires_grad=True)}
    assert belief_consistency(a,a)==0
    loss=belief_consistency(a,b);assert loss>0;loss.backward()
    assert all(x.grad is not None for o in (a,b) for x in o.values())


def test_evaluation_preserves_rng_and_weights():
    m=make_model();state={k:v.clone() for k,v in m.state_dict().items()}
    torch.manual_seed(44);before=torch.get_rng_state()
    isolated_evaluate(m,torch.utils.data.TensorDataset(*sample()),batch_size=3,chunk=2)
    assert torch.equal(before,torch.get_rng_state()) and m.training
    assert all(torch.equal(v,state[k]) for k,v in m.state_dict().items())


def test_presets_and_scheduler():
    base=['--data','dummy.npz','--output','dummy']
    cfg=resolved_config(parser().parse_args(base));validate_config(cfg)
    assert cfg['loss']['error_kl']==.01 and cfg['evidence_mode']=='shared'
    cfg=resolved_config(parser().parse_args(base+['--preset','v5']))
    assert cfg['loss']['error_kl']==0
    cfg=resolved_config(parser().parse_args(base+['--preset','control']))
    assert cfg['loss']['vicinal']==cfg['loss']['noise']==0
    cfg.update(scheduler='cosine',schedule_epochs=100)
    assert learning_rate(cfg,1)==cfg['lr']
    assert learning_rate(cfg,100)==pytest.approx(cfg['lr']*.1)


@pytest.mark.parametrize('mode',['shared','per_label'])
def test_cli_train_resume_select_and_explicit_test(tmp_path,mode):
    data=tmp_path/'data.npz';write_data(data)
    common=['--data',data,'--hidden',12,'--layers',2,'--batch-size',5,'--device','cpu',
            '--threads',1,'--evidence-mode',mode,'--consistency-weight',.02,'--scheduler','cosine','--schedule-epochs',4]
    run(*common,'--epochs',1,'--output',tmp_path/'first')
    first=torch.load(tmp_path/'first'/'last.pt',weights_only=True)
    assert first['epoch']==1 and 'optimizer' in first
    assert not (tmp_path/'first'/'test').exists()
    history=json.loads((tmp_path/'first'/'history.jsonl').read_text())
    assert history['train_rows']==13 and 'validation' in history
    assert first['best']['map']['split']=='val'
    run('--data',data,'--resume',tmp_path/'first'/'last.pt','--epochs',2,'--output',tmp_path/'resumed','--device','cpu','--threads',1)
    run(*common,'--epochs',2,'--output',tmp_path/'continuous')
    a=torch.load(tmp_path/'resumed'/'last.pt',weights_only=True)
    b=torch.load(tmp_path/'continuous'/'last.pt',weights_only=True)
    assert all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items())
    for k,st in a['optimizer']['state'].items():
        for name,value in st.items():
            assert torch.equal(value,b['optimizer']['state'][k][name])
    run('--data',data,'--checkpoint',tmp_path/'first'/'best_map.pt','--evaluate','test',
        '--output',tmp_path/'test_eval','--device','cpu','--threads',1)
    report=json.loads((tmp_path/'test_eval'/'metrics.json').read_text())
    assert report['retrieval']['queries']==7 and report['protocol']['no_filtering']
    assert json.loads((tmp_path/'test_eval'/'evaluation_manifest.json').read_text())['selected_by']=='val_map'


def test_validation_cadence_does_not_change_training(tmp_path):
    data=tmp_path/'data.npz';write_data(data)
    common=['--data',data,'--hidden',12,'--layers',2,'--batch-size',5,'--device','cpu','--threads',1,'--epochs',2]
    for cadence in [1,2]:run(*common,'--val-every',cadence,'--output',tmp_path/str(cadence))
    a=torch.load(tmp_path/'1'/'last.pt',weights_only=True);b=torch.load(tmp_path/'2'/'last.pt',weights_only=True)
    assert all(torch.equal(v,b['model_state'][k]) for k,v in a['model_state'].items())


def test_test_as_validation_is_rejected(tmp_path):
    data=tmp_path/'data.npz';arrays=write_data(data)
    for key in ('image','text','labels'):arrays['val_'+key]=arrays['test_'+key]
    np.savez(data,**arrays)
    p=subprocess.run([sys.executable,str(ROOT/'run_experiment.py'),'--data',str(data),'--output',str(tmp_path/'out')],capture_output=True,text=True,timeout=30)
    assert p.returncode!=0 and 'Validation equals test' in p.stderr


def test_invalid_overrides_existing_output_and_soft_targets():
    with pytest.raises(ValueError):ExperimentLoss(kl_anneal_epochs=2.5).validate()
    with pytest.raises(ValueError):ExperimentLoss(consistency=-1).validate()
    with pytest.raises(ValueError):make_model('invalid')
    with pytest.raises(ValueError):experiment_objective(make_model(),sample(),ExperimentLoss(),(float('nan'),1))


def test_sweep_generation_is_bounded_reproducible_and_does_not_launch(tmp_path):
    grid=tmp_path/'grid.json';grid.write_text(json.dumps({'lr':[1e-5,5e-5],'kl-weight':[0,.01],'evidence-mode':['shared','per_label']}))
    for folder in ('a','b'):
        p=subprocess.run([sys.executable,str(ROOT/'make_sweep.py'),'--data','dummy.npz','--root',str(tmp_path/folder),
            '--grid',str(grid),'--trials','3','--seeds','2','3','--device','cpu'],capture_output=True,text=True,timeout=30)
        assert p.returncode==0,p.stderr
        assert json.loads(p.stdout)['executed'] is False
    a=json.loads((tmp_path/'a'/'trials.json').read_text());b=json.loads((tmp_path/'b'/'trials.json').read_text())
    assert a['runs']==6 and a['configurations']==3 and a['grid_size']==8
    assert [x['config'] for x in a['trials']]==[x['config'] for x in b['trials']]
    assert not any((tmp_path/'a').glob('trial_*'))


def test_summary_uses_same_validation_epoch_and_ignores_test(tmp_path):
    root=tmp_path/'run';root.mkdir()
    c=resolved_config(parser().parse_args(['--data','d','--output','o']))
    (root/'provenance.json').write_text(json.dumps({'run_config':c}))
    (root/'validation.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        {'epoch':1,'mean_map':.5,'mean_nll':.3},{'epoch':2,'mean_map':.6,'mean_nll':.4}]))
    (root/'test').mkdir();(root/'test'/'metrics.json').write_text(json.dumps({'mean_map':.99}))
    p=subprocess.run([sys.executable,str(ROOT/'summarize_experiments.py'),'--root',str(tmp_path),'--output',str(tmp_path/'table.csv')],capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    import csv
    row=list(csv.DictReader((tmp_path/'table.csv').open()))[0]
    assert row['epoch']=='2' and row['mean_map']=='0.6' and row['mean_nll']=='0.4'


def test_evidence_zero_weight_does_not_receive_wrong_kl_gradients():
    a=make_model();data=sample();cfg=ExperimentLoss(error_kl=.01)
    torch.manual_seed(75);la,_=experiment_objective(a,data,cfg,(1,1),epoch=1);la.backward()
    b=make_model();torch.manual_seed(75);lb,_=experiment_objective(b,data,ExperimentLoss(),(1,1),epoch=1);lb.backward()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p.grad,q.grad)

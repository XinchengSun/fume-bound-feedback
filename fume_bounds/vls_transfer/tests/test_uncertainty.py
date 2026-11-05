import json
import pathlib
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from uncertainty import (SCORES,dirichlet_statistics,statistics_from_logits,
                         detection_metrics,risk_coverage,binary_report,ood_report)
from evaluate_vls import sample_pairs,evaluate_model
from vls_transfer import FUMEVLS,VLSConfig,matching_pairs,edl_expected_cross_entropy
from test_vls_transfer import TinyFUME,sample


def test_uniform_beta_analytic_values():
    s=dirichlet_statistics(torch.ones(1,2))
    assert s['total_entropy'].item()==pytest.approx(np.log(2))
    assert s['aleatoric_entropy'].item()==pytest.approx(.5)
    assert s['epistemic_mi'].item()==pytest.approx(np.log(2)-.5)
    assert s['dirichlet_entropy'].item()==pytest.approx(0)
    assert s['vacuity'].item()==1


@pytest.mark.parametrize('k',[2,3,20])
def test_entropy_agrees_with_pytorch_distribution(k):
    torch.manual_seed(k)
    a=1+torch.rand(8,k,dtype=torch.float64)*20
    s=dirichlet_statistics(a)
    assert torch.allclose(s['dirichlet_entropy'],torch.distributions.Dirichlet(a).entropy(),atol=1e-11)
    assert torch.allclose(s['total_entropy'],s['aleatoric_entropy']+s['epistemic_mi'],atol=1e-12)
    assert (s['epistemic_mi']>=0).all()


def test_conditional_entropy_agrees_with_monte_carlo():
    torch.manual_seed(4)
    a=torch.tensor([[2.,5.],[10.,10.]],dtype=torch.float64)
    draws=torch.distributions.Dirichlet(a).sample((40000,))
    mc=-(draws*draws.log()).sum(-1).mean(0)
    assert torch.allclose(mc,dirichlet_statistics(a)['aleatoric_entropy'],atol=.003)


def test_same_mean_different_concentration():
    s=dirichlet_statistics(torch.tensor([[1.,1.],[100.,100.]],dtype=torch.float64))
    assert torch.equal(s['mean_probability'][0],s['mean_probability'][1])
    assert s['epistemic_mi'][1]<s['epistemic_mi'][0]
    assert s['aleatoric_entropy'][1]>s['aleatoric_entropy'][0]
    assert s['vacuity'][1]<s['vacuity'][0]


def test_class_permutation_invariance():
    a=torch.tensor([[1.,2.,10.]])
    x,y=dirichlet_statistics(a),dirichlet_statistics(a.flip(-1))
    for k in SCORES: assert torch.allclose(x[k],y[k],atol=1e-12)


@pytest.mark.parametrize('a',[[[0.,1.]],[[float('nan'),1.]],[[float('inf'),1.]],[[1.]],[]])
def test_reject_invalid_concentrations(a):
    with pytest.raises(ValueError): dirichlet_statistics(torch.tensor(a))


def test_high_concentration_and_logits_are_finite():
    s=dirichlet_statistics(torch.tensor([[1.,1e6],[1e6,1e6]],dtype=torch.float64))
    assert all(torch.isfinite(v).all() for v in s.values())
    x=statistics_from_logits(torch.tensor([[-1000.,1000.]]))
    assert all(torch.isfinite(v).all() for v in x.values())


def test_binary_metrics_known_values():
    report,_=binary_report([0,1],[.25,.75],{'u':[.1,.1]},bins=4)
    assert report['brier_binary']==pytest.approx(.0625)
    assert report['brier_two_class_sum']==pytest.approx(.125)
    assert report['nll']==pytest.approx(-np.log(.75))
    assert report['ece_match_probability']==pytest.approx(.25)
    assert report['accuracy']==1
    assert report['uncertainty']['u']['error_detection']['auroc'] is None


def test_detection_thresholds_and_ties():
    good=detection_metrics([0,0,1,1],[0,0,1,1])
    assert good['auroc']==good['average_precision']==1
    assert good['fpr_at_tpr95']==0
    tied=detection_metrics([0,1,0,1],[.5]*4)
    assert tied['auroc']==tied['average_precision']==.5
    assert tied['fpr_at_tpr95']==1
    assert detection_metrics([1,1],[.1,.2])['auroc'] is None


def test_detection_auc_matches_pairwise_definition():
    rng=np.random.default_rng(5)
    y=np.r_[np.zeros(20),np.ones(15)]
    s=rng.integers(0,8,len(y))
    a,b=s[y==1,None],s[y==0][None,:]
    truth=np.mean((a>b)+.5*(a==b))
    assert detection_metrics(y,s)['auroc']==pytest.approx(truth)


def test_risk_coverage_ties_ignore_input_order():
    a=risk_coverage([0,0,1,1],[.2]*4)
    b=risk_coverage([1,1,0,0],[.2]*4)
    assert np.allclose(a['risk'],[.5]*4)
    assert np.array_equal(a['risk'],b['risk'])
    assert a['aurc']==.5
    assert risk_coverage([0,1],[0,1])['aurc']==.25
    assert risk_coverage([0,1],[1,0])['aurc']==.75
    assert risk_coverage([0,0],[0,1])['aurc']==0


def test_pair_sampling_unique_reproducible_and_complete():
    i,j=sample_pairs(20,50,7)
    i2,j2=sample_pairs(20,50,7)
    assert np.array_equal(i,i2) and np.array_equal(j,j2)
    assert len(set(zip(i,j)))==50
    i,j=sample_pairs(3,0,8)
    assert set(zip(i,j))=={(a,b) for a in range(3) for b in range(3)}
    with pytest.raises(ValueError): sample_pairs(3,-1,1)


@pytest.mark.parametrize('cfg',[VLSConfig(noise_std=float('nan')),
                                VLSConfig(clean_weight=float('inf')),
                                VLSConfig(negatives_per_anchor=1.5)])
def test_config_rejects_nonfinite_or_noninteger(cfg):
    with pytest.raises(ValueError): cfg.validate()


def test_empty_or_soft_training_labels_rejected():
    for y in (torch.tensor([[0,0],[1,0]]),torch.tensor([[.5,.5],[1,0]])):
        with pytest.raises(ValueError): matching_pairs(y)
    with pytest.raises(ValueError): edl_expected_cross_entropy(torch.zeros(1,2),torch.tensor([1.2]))


def test_evaluation_does_not_mutate_model_and_exports_all_pairs(tmp_path):
    x,t,y=sample()
    model=FUMEVLS(TinyFUME(),3)
    state={k:v.clone() for k,v in model.state_dict().items()}
    loader=DataLoader(TensorDataset(x,t,y),batch_size=5)
    r=evaluate_model(model,loader,'cpu',tmp_path,pair_budget=0,chunk=7,retrieval_chunk=4)
    assert model.training
    assert r['uncertainty']['n_pairs']==len(y)**2
    assert r['ood_status']=='not_run_no_ood_data'
    for k,v in model.state_dict().items(): assert torch.equal(v,state[k])
    with np.load(tmp_path/'pair_predictions.npz') as p:
        assert len(p['alpha'])==144
        assert np.allclose(p['total_entropy'],p['aleatoric_entropy']+p['epistemic_mi'])
        assert np.allclose(p['mean_probability'].sum(1),1)
    with pytest.raises(FileExistsError): evaluate_model(model,loader,'cpu',tmp_path)
    assert len(list(tmp_path.glob('risk_coverage_*.csv')))==5
    json.loads((tmp_path/'evaluation_summary.json').read_text())


def test_ood_is_separate_from_nonmatch_detection():
    ids={k:np.array([.1,.2]) for k in SCORES}
    oods={k:np.array([.8,.9]) for k in SCORES}
    r=ood_report(ids,oods)
    assert r['positive_class']=='OOD'
    assert all(x['auroc']==1 for x in r['scores'].values())


def _mock_repo(path):
    path.mkdir()
    (path/'model.py').write_text('''import torch
from torch import nn
class FUME(nn.Module):
 def __init__(self,device,img_input_dim,text_input_dim,output_dim,num_class,layer_num):
  super().__init__(); self.img_net=nn.Linear(img_input_dim,output_dim); self.text_net=nn.Linear(text_input_dim,output_dim); self.W=torch.eye(output_dim,num_class).to(device)
 def forward(self,i,t):
  a=self.img_net(i); b=self.text_net(t)
  return dict(view1_feature=a,view2_feature=b,view1_membershipDegree=a.softmax(-1),view2_membershipDegree=b.softmax(-1))
''')
    (path/'processor.py').write_text('''def calc_loss(a,b,pa,pb,y1,y2,w):
 return ((pa-y1)**2).mean()+((pb-y2)**2).mean()+w*((a-b)**2).mean()
''')
    (path/'options.py').write_text('''import torch
from torch.utils.data import TensorDataset,DataLoader
def get_dataloader(name):
 torch.manual_seed(42)
 def make(n):
  return DataLoader(TensorDataset(torch.randn(n,4),torch.randn(n,5),torch.nn.functional.one_hot(torch.arange(n)%3,3)),batch_size=6)
 return dict(input_dim_I=4,input_dim_T=5,class_number=3,layer_num=1,alpha=.05),dict(train=make(24),test=make(12))
''')


def _run(script,args):
    p=subprocess.run([sys.executable,str(ROOT/script),*map(str,args)],text=True,capture_output=True,timeout=40)
    assert p.returncode==0,p.stdout+p.stderr
    return p


def test_train_reload_same_evidence_predictions_and_ood_interface(tmp_path):
    up=tmp_path/'up';_mock_repo(up)
    run=tmp_path/'train'
    _run('train_fume_vls.py',['--fume-dir',up,'--output',run,'--epochs',1,'--batch-size',6,
                             '--device','cpu','--threads',1,'--uq-pairs',0,'--uq-chunk',7])
    cp=torch.load(run/'vls_final.pt',weights_only=True)
    assert cp['format']=='fume_vls_transfer_v2' and cp['evidence_head_trained']
    out=tmp_path/'reload'
    np.savez(tmp_path/'synthetic_ood.npz',image_features=np.ones((8,4)),text_features=np.ones((8,5)))
    _run('evaluate_vls.py',['--fume-dir',up,'--checkpoint',run/'vls_final.pt','--output',out,
                           '--device','cpu','--threads',1,'--uq-pairs',0,'--uq-chunk',5,
                           '--ood-pairs',tmp_path/'synthetic_ood.npz'])
    with np.load(run/'pair_predictions.npz') as a,np.load(out/'pair_predictions.npz') as b:
        for k in a.files: assert np.allclose(a[k],b[k],atol=1e-6),k
    assert json.loads((run/'test_metrics.json').read_text())==json.loads((out/'test_metrics.json').read_text())
    assert json.loads((out/'ood_report.json').read_text())['n_ood']==8
    assert (out/'checkpoint_manifest.json').exists()


def test_disabled_vls_does_not_report_random_head_uncertainty(tmp_path):
    up=tmp_path/'up';_mock_repo(up)
    run=tmp_path/'baseline'
    _run('train_fume_vls.py',['--fume-dir',up,'--output',run,'--epochs',1,'--batch-size',6,
                             '--device','cpu','--threads',1,'--vls-weight',0])
    assert not (run/'pair_predictions.npz').exists()
    assert json.loads((run/'evaluation_summary.json').read_text())['uq_status']=='not_run_evidence_head_not_trained'


def test_evaluation_rejects_encoder_only_checkpoint(tmp_path):
    cp=tmp_path/'encoder.pt';torch.save({'state_dict':{},'fixed_W':torch.ones(1)},cp)
    p=subprocess.run([sys.executable,str(ROOT/'evaluate_vls.py'),'--fume-dir',str(tmp_path),
                      '--checkpoint',str(cp),'--output',str(tmp_path/'out'),'--device','cpu'],
                      text=True,capture_output=True,timeout=20)
    assert p.returncode!=0 and 'NO trained VLS head' in p.stderr


def test_expected_ce_gradient_matches_formula_51():
    a=torch.tensor([[2.,4.],[5.,3.]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[.7,.3],[.2,.8]],dtype=torch.float64)
    loss=(t*(torch.digamma(a.sum(-1,keepdim=True))-torch.digamma(a))).sum()
    grad=torch.autograd.grad(loss,a)[0]
    expected=torch.polygamma(1,a.sum(-1,keepdim=True))-t*torch.polygamma(1,a)
    assert torch.allclose(grad,expected,atol=1e-12)


def test_beta_mixing_second_moment_coefficient():
    beta=torch.tensor(10.,dtype=torch.float64)
    d=torch.distributions.Beta(beta,beta)
    factor=2*(d.variance+d.mean.square())
    assert factor.item()==pytest.approx(11/21)
    # This checks variance algebra, not a test-risk or mAP guarantee.

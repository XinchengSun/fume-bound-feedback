import copy
import hashlib
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
from model import ModelConfig
from experiment import ExperimentLoss, experiment_objective
from evidence_ab import (build_model, PairConfig, sample_training_pairs, ab_objective, MatchingModel)
from ab_metrics import evaluate_ab, uniform_eval_pairs, report_summary
from run_evidence_ab import backbone_hash

torch.set_num_threads(1)


def tiny(task='category'):
    torch.manual_seed(47)
    return build_model(ModelConfig(5,6,4,hidden=12,layers=2),task,pair_hidden=8)


def batch(n=7):
    g=torch.Generator().manual_seed(21)
    y=F.one_hot(torch.arange(n)%4,4).float()
    if n>=2:y[0]=torch.tensor([1.,1.,0.,0.])
    return torch.randn(n,5,generator=g),torch.randn(n,6,generator=g),y


def test_matching_replaces_category_not_adds_it():
    a,b=tiny(),tiny('matching')
    assert backbone_hash(a)==backbone_hash(b)
    keys=b.state_dict().keys()
    assert not any(k.startswith(('img_evidence','text_evidence')) for k in keys)
    assert any(k.startswith('pair_head.') for k in keys)
    for o in b(*batch()[:2]):
        assert 'alpha' not in o and 'beta' not in o
        assert o['probability'].shape==(7,4)
    assert not any('pair_head' in k for k in a.state_dict())


def test_task_head_init_does_not_change_data_rng():
    tiny();a=torch.rand(10)
    tiny('matching');b=torch.rand(10)
    assert torch.equal(a,b)


@pytest.mark.parametrize('positive_mode',['overlap','aligned'])
def test_pair_labels_come_from_complete_semantic_intersection(positive_mode):
    y=batch()[2];torch.manual_seed(7)
    i,j,t,logs=sample_training_pairs(y,PairConfig(2,positive_mode,3))
    overlap=(y@y.T)>0
    assert torch.equal(overlap[i,j],t[:,0].bool())
    assert logs['positives']==len(y) and logs['negatives']==2*len(y)
    if positive_mode=='aligned':assert torch.equal(i[:len(y)],j[:len(y)])
    else:
        found=False
        for seed in range(20):
            torch.manual_seed(seed)
            ii,jj,_,_=sample_training_pairs(y)
            found|=bool((ii[:len(y)]!=jj[:len(y)]).any())
        assert found


def test_no_disjoint_labels_do_not_create_false_negatives_or_drop_rows():
    y=torch.tensor([[1.,0.],[1.,1.],[1.,0.]])
    i,j,t,logs=sample_training_pairs(y)
    assert len(t)==3 and t.eq(1).all() and logs['anchors_without_negative']==3
    assert logs['negatives']==0


def test_singleton_and_empty_labels():
    i,j,t,logs=sample_training_pairs(torch.tensor([[0.,1.]]))
    assert len(t)==1 and logs['negatives']==0
    with pytest.raises(ValueError,match='Empty'):sample_training_pairs(torch.zeros(2,4))


def test_pair_head_has_unit_prior_and_no_labels_as_inputs():
    m=tiny('matching');x,t,y=batch()
    oi,ot=m(x,t);o=m.pair_head(oi['z'],ot['z'])
    assert o['alpha'].shape==(7,1) and (o['alpha']>=1).all() and (o['beta']>=1).all()
    assert torch.allclose(o['probability'],o['predictive_logits'].sigmoid())
    assert torch.allclose(m.matching(x,t)['alpha'],o['alpha'])
    with torch.no_grad():
        m.pair_head.net[-1].weight.zero_();m.pair_head.net[-1].bias.fill_(-1000)
    o=m.matching(x,t)
    assert torch.equal(o['alpha'],torch.ones(7,1))
    assert torch.equal(o['probability'],torch.full((7,1),.5))


def test_category_posterior_is_exact_legacy_objective():
    a=tiny();b=copy.deepcopy(a);cfg=ExperimentLoss(error_kl=.01)
    torch.manual_seed(49);la,_=experiment_objective(a,batch(),cfg,(.7,1.2),epoch=4,backward=True)
    torch.manual_seed(49);lb,_=ab_objective(b,batch(),cfg,(.7,1.2),epoch=4,backward=True,classification_source='posterior')
    assert torch.equal(la,lb)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p.grad,q.grad)


def test_common_classifier_and_contrastive_exactly_equal_before_evidence():
    a,b=tiny(),tiny('matching');cfg=ExperimentLoss(vicinal=0,noise=0)
    la,_=ab_objective(a,batch(),cfg,(1.,1.),backward=True)
    lb,_=ab_objective(b,batch(),cfg,(1.,1.),backward=True)
    assert torch.equal(la,lb)
    for n,p in a.named_parameters():
        if n.startswith(('img_net.','text_net.')):assert torch.equal(p.grad,dict(b.named_parameters())[n].grad)


@pytest.mark.parametrize('task',['category','matching'])
def test_joint_and_staged_backward_agree(task):
    a=tiny(task);b=copy.deepcopy(a);cfg=ExperimentLoss(error_kl=.03)
    pc=PairConfig(chunk=3,clean_weight=.2 if task=='matching' else 0.)
    torch.manual_seed(71);la,_=ab_objective(a,batch(),cfg,(.7,1.2),epoch=4,pair_config=pc);la.backward()
    torch.manual_seed(71);lb,_=ab_objective(b,batch(),cfg,(.7,1.2),epoch=4,pair_config=pc,backward=True)
    assert torch.allclose(la,lb,atol=1e-7)
    for (n,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert p.grad is not None,n
        assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=2e-5),n
    w=b.W.clone();before={n:p.detach().clone() for n,p in b.named_parameters()}
    torch.optim.Adam(b.parameters(),lr=1e-3).step()
    for prefix in (('img_net.','text_net.','pair_head.') if task=='matching' else ('img_net.','text_net.','img_evidence.','text_evidence.')):
        assert any(not torch.equal(before[n],p) for n,p in b.named_parameters() if n.startswith(prefix))
    assert torch.equal(w,b.W)


@pytest.mark.parametrize('branch',['vicinal','noise'])
def test_each_matching_virtual_path_alone_trains_towers_and_pair_head(branch):
    m=tiny('matching')
    cfg=ExperimentLoss(classification=0,contrastive=0,vicinal=float(branch=='vicinal'),noise=float(branch=='noise'))
    l,logs=ab_objective(m,batch(),cfg,(.7,1.2),pair_config=PairConfig(chunk=3),backward=True)
    assert torch.isfinite(l) and branch in logs
    for prefix in ('img_net.','text_net.','pair_head.'):
        assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if n.startswith(prefix))


def test_pair_chunk_weighted_average_matches_full_batch():
    a=tiny('matching');b=copy.deepcopy(a);cfg=ExperimentLoss()
    torch.manual_seed(22);la,_=ab_objective(a,batch(),cfg,(1.,1.),pair_config=PairConfig(chunk=3));la.backward()
    torch.manual_seed(22);lb,_=ab_objective(b,batch(),cfg,(1.,1.),pair_config=PairConfig(chunk=100));lb.backward()
    assert torch.allclose(la,lb,atol=1e-7)
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=3e-7,rtol=2e-5)


def test_matching_rejects_undefined_category_options():
    m=tiny('matching')
    with pytest.raises(ValueError,match='posterior'):ab_objective(m,batch(),ExperimentLoss(),(1,1),classification_source='posterior')
    with pytest.raises(ValueError,match='consistency'):ab_objective(m,batch(),ExperimentLoss(consistency=.01),(1,1))
    with pytest.raises(ValueError,match='per_label'):build_model(ModelConfig(5,6,4,12,2),'matching','per_label')


@pytest.mark.parametrize('task',['category','matching'])
def test_evaluation_reports_correct_target_preserves_state(task,tmp_path):
    m=tiny(task);state=copy.deepcopy(m.state_dict());ds=torch.utils.data.TensorDataset(*batch())
    r=evaluate_ab(m,ds,tmp_path/'eval',batch_size=3,chunk=2,pair_budget=0,pair_chunk=5)
    assert m.training and r['retrieval']['queries']==7
    assert not r['retrieval']['confidence_filter']
    assert 'common_nll' in report_summary(r)
    for k in state:assert torch.equal(state[k],m.state_dict()[k])
    if task=='matching':
        q=r['native_uncertainty']['matching'];assert q['n_pairs']==49
        assert set(r['native_uncertainty'])=={'matching'}
        with np.load(tmp_path/'eval'/'matching_uq'/'predictions.npz') as f:
            y=ds.tensors[2].numpy();truth=((y[f['image_index']]>0)&(y[f['text_index']]>0)).any(1)
            assert np.array_equal(truth,f['labels'][:,0])
            assert np.allclose(f['total_entropy'],f['aleatoric_entropy']+f['epistemic_mi'])
    else:assert set(r['native_uncertainty'])=={'image','text'}
    with pytest.raises(FileExistsError):evaluate_ab(m,ds,tmp_path/'eval')


def test_untrained_pair_head_not_reported_as_uq():
    r=evaluate_ab(tiny('matching'),torch.utils.data.TensorDataset(*batch()),uq_enabled=False)
    assert r['native_uncertainty']=={}
    assert r['protocol']['uq_status'].startswith('not_run')


def test_eval_sampling_fixed_uniform_and_complete():
    i,j=uniform_eval_pairs(13,100,3);ii,jj=uniform_eval_pairs(13,100,3)
    assert np.array_equal(i,ii) and np.array_equal(j,jj) and len(set(zip(i,j)))==100
    i,j=uniform_eval_pairs(3,0,7);assert set(zip(i,j))=={(a,b) for a in range(3) for b in range(3)}


def write_data(path):
    rng=np.random.default_rng(3);a={}
    for split,n in [('train',13),('val',6),('test',7)]:
        a[split+'_image']=rng.normal(size=(n,5)).astype(np.float32)
        a[split+'_text']=rng.normal(size=(n,6)).astype(np.float32)
        y=np.eye(4,dtype=np.float32)[np.arange(n)%4];y[0]=[1,1,0,0]
        a[split+'_labels']=y
    np.savez(path,**a)


def cli(*args,ok=True):
    r=subprocess.run([sys.executable,str(ROOT/'run_evidence_ab.py'),*map(str,args)],capture_output=True,text=True,timeout=40)
    if ok:assert r.returncode==0,r.stdout+r.stderr
    else:assert r.returncode!=0
    return r


@pytest.mark.parametrize('task',['category','matching'])
def test_cli_train_resume_and_independent_evaluation(task,tmp_path):
    data=tmp_path/'data.npz';write_data(data)
    common=['--data',data,'--device','cpu','--threads',1]
    train=['--hidden',12,'--layers',2,'--batch-size',5,'--evidence-task',task,
           '--classification-source','direction','--pair-hidden',8,'--pair-chunk',3,'--uq-pairs',0]
    cli(*common,*train,'--epochs',1,'--output',tmp_path/'one')
    cp=torch.load(tmp_path/'one'/'last.pt',weights_only=True)
    assert cp['epoch']==1 and cp['run_config']['evidence_task']==task
    assert json.loads((tmp_path/'one'/'history.jsonl').read_text())['train_rows']==13
    cli(*common,'--evaluate','val','--checkpoint',tmp_path/'one'/'best_map.pt','--output',tmp_path/'eval1')
    cli(*common,'--evaluate','val','--checkpoint',tmp_path/'one'/'best_map.pt','--output',tmp_path/'eval2')
    assert json.loads((tmp_path/'eval1'/'metrics.json').read_text())==json.loads((tmp_path/'eval2'/'metrics.json').read_text())
    assert not (tmp_path/'one'/'test').exists()
    cli(*common,'--resume',tmp_path/'one'/'last.pt','--epochs',2,'--output',tmp_path/'resume')
    cli(*common,*train,'--epochs',2,'--output',tmp_path/'two')
    a=torch.load(tmp_path/'resume'/'last.pt',weights_only=True)
    b=torch.load(tmp_path/'two'/'last.pt',weights_only=True)
    for k in a['model_state']:assert torch.equal(a['model_state'][k],b['model_state'][k]),k
    assert torch.equal(a['loader_rng'],b['loader_rng'])
    cli(*common,'--resume',tmp_path/'one'/'last.pt','--epochs',2,'--evidence-task',task,'--output',tmp_path/'bad',ok=False)


def test_cli_legacy_checkpoints_and_overwrites_rejected(tmp_path):
    cp=tmp_path/'legacy.pt';torch.save({'format':'fume_v5_experiment_v1'},cp)
    data=tmp_path/'d.npz';write_data(data)
    r=cli('--data',data,'--checkpoint',cp,'--evaluate','test','--output',tmp_path/'e',ok=False)
    assert 'Not an evidence A/B checkpoint' in r.stderr
    existing=tmp_path/'exists';existing.mkdir();(existing/'keep').write_text('keep')
    cli('--data',data,'--output',existing,ok=False)
    assert (existing/'keep').read_text()=='keep'


def test_reused_modules_are_byte_identical_to_pinned_v5():
    blobs={'model.py':'4f6c761eccc038b048c38a77b63633faed1ba179',
           'losses.py':'261e8ea2b99066264b1537d185ef81674c9443f0',
           'experiment.py':'e3906e1630f795f50ddf3ffbc20e7f5901207f0b',
           'metrics.py':'330eee718b5364353d0f63b9eab33a5994c4623a',
           'data.py':'533956cca15e0fd7fdf417915f92b949e381b809'}
    for path,want in blobs.items():
        b=(ROOT/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==want,path


def test_comparison_script_runs_both_arms_and_writes_common_summary(tmp_path):
    data=tmp_path/'data.npz';write_data(data)
    output=tmp_path/'ab'
    r=subprocess.run(['bash',str(ROOT/'compare_evidence.sh'),str(data),str(output),
        '--device','cpu','--epochs','1','--hidden','12','--layers','2',
        '--batch-size','5','--pair-hidden','8','--threads','1','--uq-pairs','0'],
        capture_output=True,text=True,timeout=60)
    assert r.returncode==0,r.stdout+r.stderr
    assert (output/'comparison.csv').exists()
    pa=json.loads((output/'category'/'provenance.json').read_text())
    pb=json.loads((output/'matching'/'provenance.json').read_text())
    assert pa['backbone_sha256']==pb['backbone_sha256']
    assert pa['run_config']['classification_source']==pb['run_config']['classification_source']=='direction'
    import csv
    rows=list(csv.DictReader((output/'comparison.csv').open()))
    assert len(rows)==2 and {r['evidence_task'] for r in rows}=={'category','matching'}
    assert all(r['epoch']=='1' for r in rows)


def test_script_rejects_overriding_both_tasks(tmp_path):
    r=subprocess.run(['bash',str(ROOT/'compare_evidence.sh'),'not_read.npz',str(tmp_path/'new'),
                      '--evidence-task','matching'],capture_output=True,text=True)
    assert r.returncode!=0 and 'Do not override' in r.stderr
    assert not (tmp_path/'new').exists()

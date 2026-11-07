"""Strict read-only audit of a completed all-weights-one matching experiment.

Uses saved JSONL/JSON from run.py; does not load or alter checkpoints or select epochs.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def json_file(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    items=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
    if not items:
        raise ValueError(f'Empty log: {path}')
    return items


def assert_close(a,b,name,atol=1e-5):
    if not math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=atol):
        raise AssertionError(f'{name}: {a} != {b}')


def epochs(values,name):
    if len(values)!=150 or [r.get('epoch') for r in values]!=list(range(1,151)):
        raise AssertionError(f'{name}: expected 150 consecutive epochs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--reference-order',required=True,type=Path)
    args=p.parse_args()
    folder=args.run
    provenance=json_file(folder/'provenance.json')
    cfg=provenance['config']
    expected={'task':'matching','classes':100,'label_base':0,'seed':2,
              'batch_size':300,'layers':4,'lr':5e-5,
              'cl_weight':1,'pair_clean_weight':1,
              'vicinal':1,'noise':1,'error_kl':0,
              'pair_negatives':1,'pair_positive_mode':'overlap',
              'pair_hidden':128,'pair_chunk':128,
              'mix_beta':10,'noise_beta':1,'noise_multiplier':1,
              'initial_evidence':2,'require_ab150':True}
    for key,v in expected.items():
        if key not in cfg: raise AssertionError('Missing config '+key)
        if type(v) in (int,float): assert_close(cfg[key],v,key,atol=1e-12)
        elif cfg[key]!=v: raise AssertionError(f'Bad {key}: {cfg[key]} != {v}')
    if provenance.get('fixed_E_membership_supervision') is not True:
        raise AssertionError('E membership supervision must be unchanged')
    if provenance.get('fixed_E_clean_objective') is not False:
        raise AssertionError('CL coefficient changed; must not claim identical total E loss')
    for key,expected_value in [('bce',False),('necessity',False),
                               ('confidence_filter',False),('unit_prior',True)]:
        if provenance.get(key) is not expected_value:
            raise AssertionError('Unexpected '+key)
    assert_close(provenance['contrastive_weight'],1,'contrastive_weight')

    history=jsonl(folder/'history.jsonl')
    validation=jsonl(folder/'validation.jsonl')
    order=jsonl(folder/'order.jsonl')
    gradients=jsonl(folder/'gradient_probe.jsonl')
    ref=jsonl(args.reference_order)
    for name,series in [('history',history),('validation',validation),
                        ('order',order),('gradient_probe',gradients)]:
        epochs(series,name)
    if len(ref)<150:
        raise AssertionError('Reference sample orders incomplete')
    for index,row in enumerate(order):
        if row!=ref[index]:
            raise AssertionError('Different original sample permutation at epoch '+str(index+1))

    for h,v,g in zip(history,validation,gradients):
        epoch=h['epoch']
        if h['train_rows']!=9000 or h['updates']!=30:
            raise AssertionError(f'Epoch {epoch}: update or sample count mismatch')
        for name in ('supervision','contrastive','weighted_contrastive','pair_clean',
                     'weighted_pair_clean','vicinal','weighted_vicinal',
                     'noise','weighted_noise','h_norm_image','h_norm_text','total','lr','seconds'):
            if name not in h or not math.isfinite(float(h[name])):
                raise AssertionError(f'Epoch {epoch}: invalid {name}')
        assert_close(h['weighted_contrastive'],h['contrastive'],'CL=1',atol=2e-5)
        assert_close(h['weighted_pair_clean'],h['pair_clean'],'Pair=1',atol=2e-5)
        assert_close(h['weighted_vicinal'],h['vicinal'],'Mix=1',atol=2e-5)
        assert_close(h['weighted_noise'],h['noise'],'Noise=1',atol=2e-5)
        assert_close(h['kl_coefficient'],0,'KL=0',atol=1e-12)
        if abs(float(h.get('weighted_error_kl',0)))>1e-12:
            raise AssertionError('KL still active')
        expected_total=(h['supervision']+h['contrastive']+
                        h['pair_clean']+h['vicinal']+h['noise'])
        assert_close(h['total'],expected_total,'five-term L_total',atol=2e-4)
        assert_close(h['lr'],5e-5,'learning rate',atol=1e-12)
        if v['queries']!=1332 or v['candidates']!=1332 or v.get('confidence_filter') is not False:
            raise AssertionError('Validation protocol mismatch')
        if g.get('batch')!=1 or g.get('output_layer_only') is not True:
            raise AssertionError('Gradient source mismatch')
        for modality in ('image','text'):
            for term in ('supervision','contrastive','pair_clean','vicinal','noise'):
                cell=g.get(modality,{}).get(term,{})
                if not math.isfinite(float(cell.get('weighted_grad_norm',float('nan')))):
                    raise AssertionError(f'Missing {modality} {term} gradient')

    complete=json_file(folder/'completion.json')
    if (complete.get('completed_epochs')!=150 or complete.get('total_updates')!=4500
            or complete.get('test_evaluated') is not True or complete.get('task')!='matching'):
        raise AssertionError('Incomplete final training')
    t=json_file(folder/'test_metrics.json')
    if t.get('queries')!=4366 or t.get('candidates')!=4366 or t.get('confidence_filter') is not False:
        raise AssertionError('Test must be all 4366 queries and candidates')
    scores={key:100*float(t[key]) for key in ('i2t_map_all','t2i_map_all','mean_map')}
    assert_close(scores['mean_map'],(scores['i2t_map_all']+scores['t2i_map_all'])/2,
                 'bidirectional mean',atol=1e-4)
    baselines={'historical_E_CL005':56.1806,
               'historical_matching_CL005_Pair0_KL001':54.7287}
    report=dict(status='verified',epochs=150,updates=4500,
                checkpoint_epoch=150,full_test_queries=4366,metric='cosine all-query mAP',
                weights=dict(CL=1,Cls=1,Ed_E=1,Ed_Mix=1,Ed_Noise=1,KL=0),
                measured_mAP_pct=scores,training_seconds=sum(r['seconds'] for r in history),
                historical_baselines_pct=baselines,
                delta_vs_historical_E_pp=scores['mean_map']-baselines['historical_E_CL005'],
                delta_vs_historical_matching_pp=scores['mean_map']-baselines['historical_matching_CL005_Pair0_KL001'],
                comparison_caveat='Both historical references used CL=.05; not a single-factor estimate of evidence value.',
                source_hashes=provenance['source_sha256'])
    (folder/'full_unity_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,
                                                                 allow_nan=False)+'\n',encoding='utf-8')
    with (folder/'full_unity_raw_epochs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f)
        writer.writerow(['epoch','supervision_cls','contrastive_raw','pair_evidence_E',
                         'mix_evidence','noise_evidence','KL_coefficient','total',
                         'h_norm_image','h_norm_text','train_seconds','lr',
                         'val_i2t','val_t2i','val_mean'])
        for h,v in zip(history,validation):
            writer.writerow([h['epoch'],h['supervision'],h['contrastive'],h['pair_clean'],
                             h['vicinal'],h['noise'],h['kl_coefficient'],h['total'],
                             h['h_norm_image'],h['h_norm_text'],h['seconds'],h['lr'],
                             v['i2t_map_all'],v['t2i_map_all'],v['mean_map']])
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

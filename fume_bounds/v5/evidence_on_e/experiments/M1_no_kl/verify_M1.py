"""Read-only verification and unsmoothed M1 CSV from one finished E+matching (KL=0) run.

No model re-evaluation, no checkpoint mutation, and no test-based model selection.
"""
import argparse
import csv
import json
import math
from pathlib import Path


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    result = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not result:
        raise ValueError('Empty JSONL: ' + str(path))
    return result


def near(actual, expected, label, atol=1e-10):
    if not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=atol):
        raise ValueError(f'{label}: expected {expected}, got {actual}')


def check_sequence(data, name, count=150):
    if len(data) != count or [row['epoch'] for row in data] != list(range(1,count+1)):
        raise ValueError(f'{name}: expected exactly {count} consecutive epochs')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--reference-order',required=True,type=Path)
    a=ap.parse_args()
    run=a.run
    provenance=read_json(run/'provenance.json')
    cfg=provenance['config']
    expected={'task':'matching','seed':2,'batch_size':300,'layers':4,'classes':100,
              'label_base':0,'vicinal':1,'noise':1,'error_kl':0,
              'pair_clean_weight':0,'pair_negatives':1,'pair_hidden':128,
              'pair_positive_mode':'overlap','pair_chunk':128,
              'mix_beta':10,'noise_beta':1,'noise_multiplier':1,
              'lr':5e-5,'initial_evidence':2,'require_ab150':True}
    for key, value in expected.items():
        if key not in cfg:
            raise ValueError('Missing configuration: '+key)
        if isinstance(value, (int,float)) and not isinstance(value,bool):
            near(cfg[key],value,key)
        elif cfg[key]!=value:
            raise ValueError(f'Config {key}: expected {value}, got {cfg[key]}')
    if provenance.get('fixed_E_clean_objective') is not True or provenance.get('bce') is not False:
        raise ValueError('Not a verified E clean objective')
    if provenance.get('confidence_filter') is not False or provenance.get('necessity') is not False:
        raise ValueError('Unexpected filtering or necessity')

    history=rows(run/'history.jsonl')
    validation=rows(run/'validation.jsonl')
    order=rows(run/'order.jsonl')
    gradients=rows(run/'gradient_probe.jsonl')
    reference=rows(a.reference_order)
    for name, series in [('history',history),('validation',validation),
                         ('order',order),('gradient_probe',gradients)]:
        check_sequence(series,name)
    if len(reference)<150:
        raise ValueError('Reference order has fewer than 150 epochs')
    for idx,record in enumerate(order):
        if record != reference[idx]:
            raise ValueError('Order mismatch at epoch '+str(idx+1))
    for record in history:
        if record['train_rows']!=9000 or record['updates']!=30:
            raise ValueError('Training count mismatch at epoch '+str(record['epoch']))
        near(record['kl_coefficient'],0,'KL coefficient')
        if 'error_kl' in record and record['error_kl']!=0:
            raise ValueError('Unexpected active KL loss')
        if 'pair_clean' in record and record['pair_clean']!=0:
            raise ValueError('Unexpected clean-pair evidence loss')
        for key in ('supervision','contrastive','weighted_contrastive','vicinal',
                    'noise','total','h_norm_image','h_norm_text','seconds','lr'):
            if key not in record or not math.isfinite(float(record[key])):
                raise ValueError('Missing/nonfinite '+key)
        near(record['weighted_contrastive'],0.05*record['contrastive'], 'CL weight',atol=2e-6)
    for record in validation:
        if record['queries']!=1332 or record['candidates']!=1332 or record['confidence_filter'] is not False:
            raise ValueError('Validation selection mismatch')
    for record in gradients:
        if record['batch']!=1 or record.get('output_layer_only') is not True:
            raise ValueError('Gradient provenance invalid')
        for side in ('image','text'):
            if side not in record:
                raise ValueError('Missing '+side+' gradient')
    completed=read_json(run/'completion.json')
    if completed['completed_epochs']!=150 or completed['total_updates']!=4500 or completed['test_evaluated'] is not True:
        raise ValueError('Not a completed fixed-budget final-test run')
    test=read_json(run/'test_metrics.json')
    if test['queries']!=4366 or test['candidates']!=4366 or test['confidence_filter'] is not False:
        raise ValueError('Test pool mismatch')
    i2t=float(test['i2t_map_all'])*100
    t2i=float(test['t2i_map_all'])*100
    mean=float(test['mean_map'])*100
    near(mean,(i2t+t2i)/2,'mean mAP',atol=2e-4)
    baseline={
        'E_seed2_historical':{'i2t':55.2771,'t2i':57.0841,'mean':56.1806},
        'E_matching_KL001_historical':{'i2t':53.6494,'t2i':55.8080,'mean':54.7287}}
    result={'status':'verified_against_saved_logs','measured_M1':{'i2t':i2t,'t2i':t2i,'mean':mean},
            'historical_user_reported_baselines':baseline,
            'delta_M1_minus_E_pp':mean-baseline['E_seed2_historical']['mean'],
            'delta_M1_minus_old_matching_pp':mean-baseline['E_matching_KL001_historical']['mean'],
            'epochs':150,'steps':4500,'test_queries_per_direction':4366,
            'sum_train_seconds':sum(float(x['seconds']) for x in history),
            'validation_final_mean':100*float(validation[-1]['mean_map']),
            'source_sha256':provenance['source_sha256'],
            'note':'Historical baselines supplied by prior GPU run; M1 read from this run. No test selection.'}
    (run/'M1_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    columns=['epoch','train_supervision','train_contrastive','train_weighted_contrastive',
             'train_mixup','train_noise','train_KL_coefficient','train_total',
             'train_h_norm_image','train_h_norm_text','train_seconds','lr',
             'validation_i2t_map','validation_t2i_map','validation_mean_map']
    with (run/'M1_raw_epochs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(columns)
        for tr,v in zip(history,validation):
            w.writerow([tr['epoch'],tr['supervision'],tr['contrastive'],tr['weighted_contrastive'],
                        tr['vicinal'],tr['noise'],tr['kl_coefficient'],tr['total'],
                        tr['h_norm_image'],tr['h_norm_text'],tr['seconds'],tr['lr'],
                        v['i2t_map_all'],v['t2i_map_all'],v['mean_map']])
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()

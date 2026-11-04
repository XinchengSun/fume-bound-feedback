"""Policy regression from the user's aggregate numbers, not checkpoint replay.

Unknown absolute bound values and per-component risks are not available.
We translate the old bound to 0 and construct an illustrative risk vector
with the supplied mean change and maximum-budget fraction. Only the decision
logic is tested. The original checkpoint/gradients/batches have NOT been read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from fume_control import BoundBracket,Evaluation,RiskGuard,decide
from dataclasses import asdict


def run():
    old=np.full(200,.001)
    delta=np.full(200,(-200*2.386e-7-8.88e-7)/199)
    delta[0]=8.88e-7
    a=Evaluation(BoundBracket(0,0,'summary-only-rule-fixture',True),tuple(old),'fixture-risks')
    b=Evaluation(BoundBracket(.0002361,.0002361,'summary-only-rule-fixture',True),tuple(old+delta),'fixture-risks')
    d=decide(a,b,RiskGuard(tuple([1e-5]*200)),min_gain=1e-8,old_dual_hint=-.0029349)
    assert d.accepted
    return {'scope':'scalar_policy_regression_NOT_real_epoch83_replay',
            'source':'user-pasted audit summary',
            'synthetic_fields':['absolute risk vector','objective zero origin','component budget vector'],
            'reported_fields':['certified gain floor 0.0002361','mean risk change -0.0000002386',
                               'maximum budget utilization 8.88%','old dual hint -0.0029349'],
            'decision':asdict(d)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--out')
    a=p.parse_args(); r=run(); text=json.dumps(r,ensure_ascii=False,indent=2)
    if a.out:
        dest=Path(a.out)
        if dest.exists(): raise FileExistsError(dest)
        dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(text,encoding='utf-8')
    print(text)

from pathlib import Path
import pandas as pd
import numpy as np

p=Path(__file__).resolve().parent
df=pd.read_csv(p/'raw.csv')
assert len(df)==178 and df.groupby(['batch','group','kind','split']).size().max()==1
assert np.isfinite(df.drop(columns=['group','kind','split','loss'])).all().all()
df.groupby(['split','group','kind']).mean(numeric_only=True).drop(columns=['batch']).to_csv(p/'averages.csv')
rows=[]
for split in ['val','test']:
    for group in ['alone','plus_FUME']:
        def values(k):
            return df[(df.split==split)&(df.kind==k)&((df.group==group)|(df.group=='base'))].set_index('batch').sort_index()
        for k in ['original','plain','map_logit','predictive_mean','stop_sd']:
            full=values('full')
            baseline=values(k)
            assert len(full)==len(baseline)==8
            for metric in ['map','brier','nll','ece','event_loss','pos_L','neg_L']:
                delta=full[metric]-baseline[metric]
                if metric=='map':delta*=100
                rows.append(dict(split=split,group=group,contrast='full_minus_'+k,metric=metric,
                         difference=delta.mean(),positive=int((delta>0).sum()),negative=int((delta<0).sum()),
                         min=delta.min(),max=delta.max(),historical_batch=delta.loc[0],seven_new=delta.loc[1:].mean()))
pd.DataFrame(rows).to_csv(p/'paired.csv',index=False)


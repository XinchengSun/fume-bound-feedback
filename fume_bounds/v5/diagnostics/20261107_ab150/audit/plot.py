import csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
fig, grid=plt.subplots(2,2,figsize=(12,7.6),layout="constrained")
a,b,c,d=grid.flat
csvrows=[]
for arm,short,color in [('A_fume','A  FUME','#363e60'),('B_bce','B  BCE','#00897b'),('C_category','C  Category evidence','#c95a2c')]:
    rows=[[json.loads(z) for z in (root/arm/file).read_text().splitlines()] for file in ('history.jsonl','validation.jsonl')]
    t,v=rows
    ep=[x['epoch'] for x in v]
    assert ep==[x['epoch'] for x in t]==list(range(1,151))
    for ax in ([a] if short.startswith('A') else [a,b]):
        ax.plot(ep,[100*x['mean_map'] for x in v],lw=1.2,marker='.',ms=2.4,c=color,label=short)
    if not short.startswith('A'):
        for ax,key in [(c,'classification'),(d,'contrastive')]:
            ax.plot(ep,[x[key] for x in t],lw=1.2,marker='.',ms=2.4,c=color,label=short)
    for tv,vv in zip(t,v):
        csvrows.append(dict(arm=arm,epoch=vv['epoch'],validation_map=vv['mean_map'],val_i2t=vv['i2t_map_all'],val_t2i=vv['t2i_map_all'],train_total=tv['total'],classification_bce=tv.get('classification'),contrastive_raw=tv.get('contrastive'),image_evidence=tv.get('evidence_image'),text_evidence=tv.get('evidence_text')))
c.set_yscale('log')
b.axhline(1.5306075,ls='--',lw=.85,c='#595959',label='Expected random rank')
c.axhline(-.01*math.log(.01)-.99*math.log(.99),ls='--',lw=.85,c='#595959',label='Constant p = .01')
d.axhline(2*math.log(599),ls='--',lw=.85,c='#595959',label='Equal logits')
for ax,title,units in [(a,'Full validation comparison','Bidirectional mAP (%)'),(b,'B and C on the same validation set','Bidirectional mAP (%)'),(c,'Clean classification objective','Mean training BCE (log scale)'),(d,'Contrastive objective before weighting','Mean training loss')]:
    ax.set(title=title,ylabel=units,xlabel='Epoch',xlim=(1,150))
    ax.grid(axis='y',alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=8)
fig.suptitle('INRIA  |  seed 2  |  one observed point per epoch',weight='bold')
fig.savefig(root/'raw_curves_ABC.png',dpi=170)
with (root/'raw_curves_ABC.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(csvrows[0])); w.writeheader(); w.writerows(csvrows)

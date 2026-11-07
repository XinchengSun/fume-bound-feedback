"""Check all final metrics and epoch rows; reconstruct test AP independently for B, C, D."""
import csv
import json
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
raw=root/"raw"
def load(path):
    return json.loads(path.read_text(encoding="utf8"))
def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]
arms={"A_fume":"A  原版 FUME", "B_bce":"B  普通 BCE", "C_category":"C  类别证据", "D_matching":"D  匹配证据"}
colors=dict(zip(arms,["#252f49","#047574","#bb5729","#7254a1"]))
protocol=load(raw/"acceptance.json")
assert load(raw/"status.json")==dict(active=None,completed=list(arms),epochs=150)
maps=load(raw/"test_map.json")
common_order=rows(raw/"A_fume"/"order.jsonl")
fields=["i2t_map_all","t2i_map_all","mean_map"]
allrows=[]
curves={}
final=[]
checks={}
paired={}
test_labels=None
for arm in arms:
    folder=raw/arm
    loss,val=rows(folder/"history.jsonl"),rows(folder/"validation.jsonl")
    assert [x["epoch"] for x in loss]==[x["epoch"] for x in val]==list(range(1,151))
    assert all(x["train_rows"]==9000 for x in loss)
    assert rows(folder/"order.jsonl")==common_order and len(common_order)==150
    csvdata=list(csv.DictReader((folder/"raw_curves.csv").open()))
    assert len(csvdata)==150
    for csvrow,t,v in zip(csvdata,loss,val):
        assert v["queries"]==v["candidates"]==1332 and not v["confidence_filter"]
        nums=[float(csvrow[k]) for k in ("epoch","train_loss","train_seconds","val_i2t_map","val_t2i_map","val_mean_map")]
        assert nums==[t["epoch"],t["total"],t["seconds"],*(v[f] for f in fields)]
        allrows.append([arm,*nums])
    curves[arm]=np.array([[x["epoch"],t["total"],*(100*x[f] for f in fields)] for x,t in zip(val,loss)])
    score=maps[arm]
    assert score["queries"]==score["candidates"]==4366 and not score["confidence_filter"]
    assert score["mean_map"]==(score["i2t_map_all"]+score["t2i_map_all"])/2
    if arm=="A_fume":
        assert score==load(folder/"test_metrics.json")
    else:
        manifest=load(folder/"test"/"evaluation_manifest.json")
        prov=load(folder/"provenance.json")
        assert prov["backbone_sha256"]==protocol["initial_backbone_and_W_sha256"]
        assert manifest["epoch"]==150 and manifest["split"]=="test" and manifest["no_confidence_filter"]
        assert manifest["data_sha256"]==protocol["data"]["file_sha256"]
        assert score==load(folder/"test"/"metrics.json")["retrieval"]
        emb=root/"embeddings"/arm/"test"
        with np.load(emb/"image_common_predictions.npz") as i,np.load(emb/"text_common_predictions.npz") as t:
            zi,zt,y=i["z"],t["z"],i["labels"]
            assert np.array_equal(y,t["labels"]) and np.all(y.sum(1)==1) and len(y)==4366
            if test_labels is None: test_labels=y
            else: assert np.array_equal(y,test_labels)
        ids=y.argmax(1)
        per=[]
        for q,g in [(zi,zt),(zt,zi)]:
            ap=[]
            for start in range(0,len(q),128):
                rank=np.argsort(cdist(q[start:start+128],g,"cosine"),axis=1,kind="stable")
                for j,rr in enumerate(rank):
                    hit=np.flatnonzero(ids[rr]==ids[start+j])+1
                    ap.append(float((np.arange(1,len(hit)+1)/hit).mean()))
            per.append(ap)
        paired[arm]=np.asarray(per)
        alt=paired[arm].mean(1)
        checks[arm]=dict(i2t=float(alt[0]),t2i=float(alt[1]),mean=float(alt.mean()))
        assert np.max(np.abs(alt-[score[f] for f in fields[:2]]))<0.00006,(arm,alt,score)
    final.append([arm,150,4366,*(100*score[f] for f in fields)])
deltas={a:{f:100*(maps["D_matching"][f]-maps[a][f]) for f in fields} for a in list(arms)[:3]}
(root/"verification.json").write_text(json.dumps(dict(independent_test_map_0_to_1=checks,D_minus_comparator_percentage_points=deltas,raw_epoch_rows=len(allrows),identical_sample_order=True,initial_backbone_sha256=protocol["initial_backbone_and_W_sha256"],validation_queries=1332,test_queries=4366),indent=2))
for name,header,contents in [
    ("four_arms_final.csv",["arm","epoch","test_queries","test_i2t_map_pct","test_t2i_map_pct","test_mean_map_pct"],final),
    ("four_arms_raw_epochs.csv",["arm","epoch","training_loss","training_seconds","val_i2t_map_0_to_1","val_t2i_map_0_to_1","val_mean_map_0_to_1"],allrows)]:
    with (root/name).open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(header);writer.writerows(contents)

plt.rcParams.update({"font.family":FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),
 "axes.unicode_minus":False,"pdf.fonttype":42,"axes.spines.right":False,"axes.spines.top":False,
 "font.size":10,"axes.edgecolor":"#b2b7c1","axes.labelcolor":"#394455","xtick.color":"#485463","ytick.color":"#485463"})
def style(ax,title,unit):
    ax.set(title=title,xlabel="训练轮次",ylabel=unit,xlim=(1,150))
    ax.set_xticks([1,30,60,90,120,150]); ax.grid(axis="y",lw=.55,alpha=.24)
    ax.title.set_fontsize(11)
def traces(ax,subset,col):
    for a in subset:
        x=curves[a];ax.plot(x[:,0],x[:,col],color=colors[a],lw=1,marker=".",ms=2.2,zorder=2)
handles=[Line2D([0],[0],color=colors[a],lw=2,label=t) for a,t in arms.items()]
fig=plt.figure(figsize=(12.5,10.6))
gs=fig.add_gridspec(3,4,height_ratios=[1.0,1.55,1.08],left=.065,right=.98,top=.905,bottom=.077,hspace=.60,wspace=.48)
fig.text(.065,.965,"INRIA  |  匹配证据与三个对照的检索结果",fontsize=20,weight="bold")
fig.text(.065,.932,"seed 2   ·   单卡串行   ·   第 150 轮权重   ·   全部 4,366 个测试查询参与余弦检索",fontsize=11,color="#475365")
tax=fig.add_subplot(gs[0,:]);tax.axis("off")
table=tax.table(cellText=[[arms[a],*(f"{x:.4f}%" for x in row[3:])] for a,row in zip(arms,final)],
 colLabels=["方案","测试图像 → 文本","测试文本 → 图像","测试双向平均"],cellLoc="center",
 colWidths=[.31,.23,.23,.23],bbox=[0,0,1,1])
for (r,c),cell in table.get_celld().items():
    cell.set_linewidth(.5);cell.set_edgecolor("#d2d7de");cell.set_facecolor("#eaf0f5" if r==0 else "#ffffff")
    if r==0 or c==3:cell.set_text_props(weight="bold")
    if r and c==0:cell.set_text_props(color=colors[list(arms)[r-1]],weight="bold")
for idx,(sub,title) in enumerate([(list(arms),"全部四组"),(list(arms)[1:],"B、C、D 局部放大")]):
    ax=fig.add_subplot(gs[1,idx*2:idx*2+2]);traces(ax,sub,4);style(ax,title,"验证集双向 mAP（%）");ax.set_ylim(bottom=0)
    ax.legend(handles=[h for h,a in zip(handles,arms) if a in sub],loc="best",frameon=True,framealpha=.88,fontsize=8,ncol=2)
for idx,a in enumerate(arms):
    ax=fig.add_subplot(gs[2,idx]);traces(ax,[a],1);style(ax,arms[a], "训练损失" if idx==0 else "")
fig.text(.065,.015,"每个点为一轮实测值。上图用验证集全部 1,332 个查询。下图为原始训练损失均值，各组目标及纵轴刻度不同。",fontsize=9,color="#445062")
fig.savefig(root/"four_arms_overview.png",dpi=160)
with PdfPages(root/"four_arms_report.pdf") as pdf:
    pdf.savefig(fig);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12.5,8.2))
    fig.subplots_adjust(top=.84,bottom=.10,hspace=.45,wspace=.26)
    fig.suptitle("两个检索方向的逐轮原始验证 mAP",fontsize=18,y=.97)
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.925),ncols=4,frameon=False)
    for row,col in enumerate([2,3]):
        for idx,(sub,label) in enumerate([(list(arms),"全部四组"),(list(arms)[1:],"B、C、D 局部放大")]):
            ax=axs[row,idx];traces(ax,sub,col);style(ax,("图像 → 文本" if row==0 else "文本 → 图像")+"  ·  "+label,"验证集 mAP（%）");ax.set_ylim(bottom=0)
    pdf.savefig(fig);plt.close(fig)
with zipfile.ZipFile(root/"INRIA_four_arms.zip","w",compression=zipfile.ZIP_DEFLATED) as bundle:
    for file in sorted(raw.rglob("*")):
        if file.is_file(): bundle.write(file,file.relative_to(root))
    for f in ("four_arms_report.pdf","four_arms_overview.png","four_arms_final.csv","four_arms_raw_epochs.csv","verification.json"):
        bundle.write(root/f,f)
print(json.dumps({"test":final,"D_minus":deltas,"independent":checks,"rows":len(allrows)},indent=1))

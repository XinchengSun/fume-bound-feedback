import csv
import json
import hashlib
import subprocess
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

root=Path(__file__).resolve().parent
old=root.parent/"fume_ab_20261107/final_150/raw"
repo=root.parent/"fume_v5_20261106/repo"
commit="bbd3b6fe66d47a65ec1b31a9b8149ce55ecda5de"
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))
def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

def main():
    h,v,g,o=[rows(root/file) for file in ["history.jsonl","validation.jsonl","gradient_probe.jsonl","order.jsonl"]]
    epochs=list(range(1,151))
    for records in (h,v,g,o):
        assert [r["epoch"] for r in records]==epochs
    assert o==rows(old/"A_fume/order.jsonl")
    acceptance=read(old/"acceptance.json")
    provenance=read(root/"provenance.json")
    assert provenance["initial_backbone_and_W_sha256"]==acceptance["initial_backbone_and_W_sha256"]
    assert provenance["data"]["file_sha256"]==acceptance["data"]["file_sha256"]
    assert provenance["config"]["require_ab150"] and provenance["final_test_requested"]
    for file,expected in provenance["source_sha256"].items():
        blob=subprocess.check_output(["git","-C",str(repo),"show",commit+":fume_bounds/v5/"+file])
        assert hashlib.sha256(blob).hexdigest()==expected
    done=read(root/"completion.json")
    assert done["test_evaluated"] and done["fixed_final_epoch"] and done["total_updates"]==4500
    for t,val,probe in zip(h,v,g):
        assert t["train_rows"]==9000 and t["lr"]==5e-5 and abs(t["total"]-t["supervision"]-t["weighted_contrastive"])<1e-5
        assert val["queries"]==val["candidates"]==1332 and not val["confidence_filter"]
        assert probe["batch"]==1 and probe["output_layer_only"]
    test=read(root/"test_metrics.json")
    assert test["queries"]==test["candidates"]==4366 and not test["confidence_filter"] and test["paired_items_included"]
    assert abs(test["mean_map"]-.5*(test["i2t_map_all"]+test["t2i_map_all"]))<1e-10
    metrics={**read(old/"test_map.json"),"E_membership":test}
    with (root/"final_test_map.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["arm","epoch","queries","i2t_map_pct","t2i_map_pct","mean_map_pct"])
        for arm,m in metrics.items():
            w.writerow([arm,150,m["queries"],*[100*m[key] for key in ("i2t_map_all","t2i_map_all","mean_map")]])
    av=rows(old/"A_fume/validation.jsonl")
    with (root/"E_raw_epochs.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f);w.writerow(["epoch","lr","train_total","train_supervision","train_contrastive","train_weighted_contrastive","val_i2t_map_pct","val_t2i_map_pct","val_mean_map_pct","A_val_mean_map_pct",*[k+"_"+s for s in ("image","text") for k in ("sup_grad_norm","weighted_CL_grad_norm","cosine")]])
        for t,val,p,a in zip(h,v,g,av):
            w.writerow([t["epoch"],t["lr"],t["total"],t["supervision"],t["contrastive"],t["weighted_contrastive"],*[100*val[k] for k in ("i2t_map_all","t2i_map_all","mean_map")],100*a["mean_map"],*[p[s][k] for s in ("image","text") for k in ("supervision_grad_norm","weighted_cl_grad_norm","cosine")]])
    plt.rcParams.update({"font.family":FontProperties(fname="C:/Windows/Fonts/msyh.ttc").get_name(),"axes.unicode_minus":False,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12.3,4.8),layout="constrained")
    for ax,series,title,ylabel in [(axes[0],[(av,"mean_map","A 原版 FUME","#36415e",100),(v,"mean_map","E 直接监督隶属度","#0a847a",100)],"固定轮次的验证表现","验证集双向 mAP（%）"),(axes[1],[(h,"total","总损失","#354262",1),(h,"supervision","隶属度监督","#0a847a",1),(h,"weighted_contrastive","对比项 × 0.05","#bd5b32",1)],"E 组训练分项","每轮训练损失均值")]:
        for data,key,label,color,factor in series:ax.plot(epochs,[factor*r[key] for r in data],lw=1.1,marker=".",ms=2.5,color=color,label=label)
        ax.set(xlabel="训练轮次",ylabel=ylabel,xlim=(1,150),title=title)
        ax.grid(axis="y",alpha=.2);ax.legend(frameon=False)
    fig.suptitle("INRIA · seed 2 · 每轮一个原始观测点",weight="bold")
    fig.savefig(root/"E_raw_curves.png",dpi=170);plt.close(fig)
    report=dict(commit=commit,epochs=150,total_updates=4500,matched_order_hashes=150,matched_initial_backbone=True,probe_batches=len(g),mean_test_map_pct=100*test["mean_map"],E_minus_A_pp=100*(test["mean_map"]-metrics["A_fume"]["mean_map"]),E_minus_B_pp=100*(test["mean_map"]-metrics["B_bce"]["mean_map"]))
    (root/"verification.json").write_text(json.dumps(report,indent=2))
    with ZipFile(root/"E_results.zip","w",compression=ZIP_DEFLATED) as archive:
        for name in ("history.jsonl","order.jsonl","gradient_probe.jsonl","validation.jsonl","initial_validation.json","provenance.json","completion.json","test_metrics.json","worker.log","pytest.log","launch_E.sh","mechanism_cpu.json","verification.json","E_raw_epochs.csv","E_raw_curves.png","final_test_map.csv"):
            archive.write(root/name,name)
    print(json.dumps(report),test)
if __name__=="__main__":
    main()

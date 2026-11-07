"""Describe class separation and readouts without tuning parameters or touching test data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, roc_auc_score

def ranking(similarity, labels):
    relevant=labels[:,None]==labels
    def direction(s):
        order=np.argsort(-s,axis=1,kind="stable")
        ordered=np.take_along_axis(relevant,order,axis=1)
        return float(((ordered.cumsum(1)*ordered/np.arange(1,len(labels)+1)).sum(1)/ordered.sum(1)).mean())
    a,b=direction(similarity),direction(similarity.T)
    return {"image_to_text":a,"text_to_image":b,"mean":(a+b)/2}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("assets",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    manifest=json.loads((Path(__file__).parent/"manifest.json").read_text())
    report={}
    q=[.05,.25,.50,.75,.95]
    for arm,meta in manifest["arms"].items():
        path=args.assets/(arm+".npz")
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta["npz_sha256"]
        with np.load(path,allow_pickle=False) as x:
            ids=x["labels"].argmax(1)
            assert (x["labels"].sum(1)==1).all()
            assert np.array_equal(x["sample_ids"],np.arange(len(ids)))
            w=x["W"].astype("float64")
            sim=1-cdist(x["z_image"],x["z_text"],"cosine")
            same=ids[:,None]==ids
            other=~same
            same[np.diag_indices(len(ids))]=False
            score=np.r_[sim[same],sim[other]]
            binary=np.r_[np.ones(same.sum(),dtype=int),np.zeros(other.sum(),dtype=int)]
            a={"cosine_quantiles_p05_p25_p50_p75_p95":{
                "same_excluding_original_pair":np.quantile(sim[same],q).tolist(),
                "different_class":np.quantile(sim[other],q).tolist()},
                "n_same_nonpaired":int(same.sum()),"n_different":int(other.sum()),
                "same_vs_different_pooled_AUROC":float(roc_auc_score(binary,score)),
                "embedding_map":ranking(sim,ids),"classification":{}}
            projected=1-cdist(x["h_image"]@w,x["h_text"]@w,"cosine")
            a["max_abs_cosine_change_h_to_hW"]=float(np.abs((1-cdist(x["h_image"],x["h_text"],"cosine"))-projected).max())
            if arm!="A_fume":
                a["raw_logits_map"]=ranking(1-cdist(x["logits_image"],x["logits_text"],"cosine"),ids)
                a["sigmoid_probability_map"]=ranking(1-cdist(x["class_probability_image"],x["class_probability_text"],"cosine"),ids)
            for n,side in enumerate(("image","text"),start=1):
                s=x["view%d_membershipDegree"%n] if arm=="A_fume" else x["class_probability_"+side]
                present=x["labels"].any(0)
                actual=s[np.arange(len(ids)),ids][:,None]
                first=1+(s>actual).sum(1)
                average_rank=first+0.5*((s==actual).sum(1)-1)
                a["classification"][side]=dict(score="membershipDegree" if arm=="A_fume" else "sigmoid(logits)",
                    present_classes=int(present.sum()),macro_AP=float(average_precision_score(x["labels"][:,present],s[:,present],average="macro")),
                    top1_with_argmax=float((s.argmax(1)==ids).mean()),mean_true_class_midrank=float(average_rank.mean()),
                    median_true_class_midrank=float(np.median(average_rank)))
            report[arm]=a
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=="__main__":
    main()

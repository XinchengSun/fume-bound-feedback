import json
from pathlib import Path
import sys
import importlib.metadata
from packaging.requirements import Requirement
import numpy as np
import torch
import scipy.io
import h5py
from torch.utils.data import TensorDataset
from compare150 import runner, ModelConfig, V5
from metrics import detection

def check(device):
    site = Path(__file__).resolve().parent/"numpy226_site"
    assert Path(np.__file__).resolve().is_relative_to(site) and np.__version__ == "2.2.6"
    versions = {}
    for line in (V5/"requirements.txt").read_text().splitlines():
        if not line.strip():
            continue
        req = Requirement(line)
        version = importlib.metadata.version(req.name)
        assert req.specifier.contains(version, prereleases=True), (line, version)
        versions[req.name] = version
    assert detection([0,1,0,1],[.1,.4,.35,.8])["auroc"] == 1.
    assert detection([0,1,1,0],[.1,.4,.35,.8])["auroc"] == .5
    torch.set_num_threads(2)
    toy = TensorDataset(torch.randn(16,8),torch.randn(16,6),torch.eye(4).repeat(4,1))
    for task in ("category","matching"):
        model = runner.build_model(ModelConfig(8,6,4,16,1),task).to(device)
        for enabled in ([False,True] if task=="category" else [True]):
            result = runner.evaluate_ab(model,toy,uq_enabled=enabled,pair_budget=64)
            assert runner.report_summary(result)["queries"]==16
    return dict(device=device,python=sys.version,versions=versions,numpy_path=np.__file__,
                upstream_eval_smoke=["B_bce","C_category","D_matching"])
if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1])))

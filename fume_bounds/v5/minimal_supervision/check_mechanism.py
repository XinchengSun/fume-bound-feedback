"""Read-only mathematical checks for the archived AB150 experiment.
These are analytic/synthetic checks, NOT a replay of the released NPZ data or training.
No network access, model-file changes, or optimizer steps occur.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.manual_seed(17)
torch.set_default_dtype(torch.float64)
k, batch = 100, 300
p = 1 / k
logit = math.log(p / (1-p))
constant_bce = -(p*math.log(p)+(1-p)*math.log1p(-p))
g = torch.full((k,), .5/k); g[0] = -.5/k
common = torch.full_like(g, g.mean())
common_fraction = float(common.square().sum()/g.square().sum())
assert abs(common_fraction - (1-2/k)**2) < 1e-12

def cl(h):
    n = len(h)//2
    z = F.normalize(h, dim=1)
    logits = z@z.T
    logits = logits.masked_fill(torch.eye(len(h), dtype=torch.bool), -torch.inf)
    target=(torch.arange(2*n)+n)%(2*n)
    return 2*F.cross_entropy(logits, target)

h = torch.randn(12, k, requires_grad=True)
g1 = torch.autograd.grad(cl(h), h)[0]
large = (h.detach()*100).requires_grad_()
g100 = torch.autograd.grad(cl(large), large)[0]
assert torch.allclose(g1/100, g100, atol=1e-14, rtol=1e-10)
collapsed = torch.full((12,k), logit, requires_grad=True)
collapse_loss=cl(collapsed)
collapse_grad = torch.autograd.grad(collapse_loss, collapsed)[0]
assert float(collapse_grad.abs().max()) < 1e-14

w, _ = torch.linalg.qr(torch.randn(k,k))
y = torch.eye(k)[torch.tensor([0,1,7,99])]
z = y@w.T
membership=(z@w).relu()
minimum_supervision_residual=float((membership-y).norm(dim=1).max())
assert minimum_supervision_residual < 1e-12

out = {
    'scope':'Analytic/synthetic checks only; released NPZ binaries were not loaded in this process.',
    'archive_commit':'6de422a322eb5b47db7b5f5be2b81c40b55c1d16',
    'training_commit':'2a8980a3ab85d35da68660db79813387016047c1',
    'constant_1_percent_prediction_reference': {
        'class_count':k,'positive_labels_per_sample':1,
        'probability_each_label':p,'logit_each_label':logit,
        'logit_vector_l2_norm':math.sqrt(k)*abs(logit),
        'mean_binary_cross_entropy':constant_bce,
        'constant_similarity_fume_cl_B300':2*math.log(2*batch-1),
    },
    'initial_zero_logits_per_sample_bce_gradient': {
        'positive_coordinate':float(g[0]),'negative_coordinate':float(g[1]),
        'common_all_coordinates_mean':float(g.mean()),
        'common_component_fraction_of_squared_norm':common_fraction,
        'note':'Gradient with respect to logits at exactly zero, not an observed encoder parameter gradient.'
    },
    'normalization_gradient_checks': {
        'norm_scaling_factor':100,
        'max_error_g100_minus_g1_div100':float((g100-g1/100).abs().max()),
        'collapsed_B6_cl':float(collapse_loss.detach()),
        'collapsed_B6_expected_cl':2*math.log(11),
        'collapsed_gradient_max_abs':float(collapse_grad.abs().max()),
        'note':'Norm scaling is an h-gradient identity with fixed directions, not a statement of parameter-gradient ratios during Adam.'
    },
    'minimal_no_necessity_onehot_geometry': {
        'max_l2_residual_relu_zW_vs_onehot':minimum_supervision_residual,
        'note':'Existence of a feasible representation only. No neural-network convergence or accuracy improvement is established.'
    },
    'quoted_not_recomputed_B_epoch3_training_log': {
        'classification':0.05640109442174435,
        'contrastive':12.79060084025065,
        'h_norm_image':46.54759585062663,
        'h_norm_text':46.14947789510091,
        'source_path':'fume_bounds/v5/diagnostics/20261107_ab150/runs/B_bce/history.jsonl:3'
    }
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, help='Optional NEW JSON; default prints only')
args = parser.parse_args()
if args.output is not None:
    with args.output.open('x', encoding='utf8') as stream:
        stream.write(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False,indent=2))
